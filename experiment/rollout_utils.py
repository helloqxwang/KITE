"""Replay batches of saved tabletop cube-pick trajectories."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment.replay_utils import _source_scalar, _trajectory_value
from experiment.tabletop_sim.env import TabletopManipulationEnv  # noqa: E402


@dataclass(frozen=True)
class RolloutResult:
    source_index: int
    source_seed: int | None
    action_frame_count: int
    completed_steps: int
    success: bool
    max_cube_z: float
    final_cube_pose: np.ndarray
    error: str


def _parse_indices(indices: object, trajectory_count: int) -> list[int]:
    if isinstance(indices, (list, tuple, ListConfig)):
        return [int(idx) for idx in indices]

    selected: list[int] = []
    for part in str(indices).split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            pieces = [int(x) if x else None for x in part.split(":")]
            if len(pieces) > 3:
                raise ValueError(f"Invalid trajectory slice {part!r}")
            selected.extend(range(*slice(*pieces).indices(int(trajectory_count))))
        else:
            selected.append(int(part))
    return selected


def _positive_int_from_cfg(cfg: DictConfig, path: str, *, default: int) -> int:
    value = OmegaConf.select(cfg, path, default=default)
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"{path} must be positive, got {value}")
    return parsed


def _validate_robot_model_q_source_matches_env(
    source: dict[str, np.ndarray],
    env: TabletopManipulationEnv,
) -> None:
    action_space_type = _source_scalar(source, "action_space_type")
    if str(action_space_type).strip() != "robot_model_q":
        return
    if str(getattr(env, "embodiment", "")) == "robotiq_gripper_stick_armless":
        expected_joint_order = (
            "gripper_x",
            "gripper_y",
            "gripper_z",
            "gripper_roll",
            "gripper_pitch",
            "gripper_yaw",
            "gripper_cmd",
            "stick_x",
            "stick_y",
            "stick_z",
            "stick_roll",
            "stick_pitch",
            "stick_yaw",
        )
        source_joint_order = tuple(
            str(x) for x in np.asarray(source.get("joint_orders", ())).reshape(-1)
        )
        if source_joint_order and source_joint_order != expected_joint_order:
            raise ValueError(
                "robot_model_q source joint order does not match the armless Robotiq+stick contract. "
                f"source={source_joint_order} env={expected_joint_order}"
            )
        return
    if not env.robot.has_group("right"):
        raise ValueError(
            "robot_model_q replay requires a right control group in the tabletop env"
        )

    side = env.robot.right
    expected_joint_order = tuple(side.arm_joint_names + side.hand_joint_names)
    source_joint_order = tuple(
        str(x) for x in np.asarray(source.get("joint_orders", ())).reshape(-1)
    )
    source_joint_order_canonical = tuple(
        _canonical_robot_model_q_joint_name(name) for name in source_joint_order
    )
    expected_joint_order_canonical = tuple(
        _canonical_robot_model_q_joint_name(name) for name in expected_joint_order
    )
    if (
        source_joint_order
        and source_joint_order_canonical != expected_joint_order_canonical
    ):
        raise ValueError(
            "robot_model_q source joint order does not match the tabletop env control order. "
            f"source={source_joint_order} env={expected_joint_order}"
        )

    robot_name = _source_scalar(source, "robot_name")
    if robot_name is None or str(robot_name).strip() == "":
        return

    from robot_model.robot_model import (
        DEFAULT_ASSETS_DIR,
        discover_robot_assets,
        resolve_robot_model_name,
    )

    asset_meta = discover_robot_assets(DEFAULT_ASSETS_DIR)
    resolved_name = resolve_robot_model_name(
        str(robot_name), assets_dir=DEFAULT_ASSETS_DIR, meta=asset_meta
    )
    resolved_meta = asset_meta.get(resolved_name)
    if resolved_meta is None or "path" not in resolved_meta:
        return
    expected_path = (
        (DEFAULT_ASSETS_DIR / str(resolved_meta["path"])).expanduser().resolve()
    )
    actual_path = Path(env.scene.robot_xml_path).expanduser().resolve()
    if (
        actual_path != expected_path
        and not _is_equivalent_robot_model_q_tabletop_asset(
            robot_name=str(robot_name),
            resolved_name=str(resolved_name),
            expected_path=expected_path,
            actual_path=actual_path,
        )
    ):
        raise ValueError(
            "robot_model_q source was generated for a different robot asset than the tabletop env compiled. "
            f"robot_name={robot_name!r} resolved={resolved_name!r} expected_asset={expected_path} "
            f"actual_asset={actual_path}"
        )


def _canonical_robot_model_q_joint_name(name: str) -> str:
    """Normalize equivalent floating-base names used by robot_model and Wuji MJCF."""

    aliases = {
        "right_pos_x": "virtual_joint_x",
        "right_pos_y": "virtual_joint_y",
        "right_pos_z": "virtual_joint_z",
        "right_rot_x": "virtual_joint_roll",
        "right_rot_y": "virtual_joint_pitch",
        "right_rot_z": "virtual_joint_yaw",
    }
    return aliases.get(str(name), str(name))


def _is_equivalent_robot_model_q_tabletop_asset(
    *,
    robot_name: str,
    resolved_name: str,
    expected_path: Path,
    actual_path: Path,
) -> bool:
    """Allow hand-authored tabletop MJCFs that preserve the robot_model_q contract."""

    robot_text = f"{robot_name} {resolved_name}".lower()
    from experiment.tabletop_sim.assets import (
        ALLEGRO_RIGHT_MJCF,
        ALLEGRO_RIGHT_URDF,
        BARRETT_MJCF,
        BARRETT_URDF,
        WUJIHAND_RIGHT_XML,
    )

    equivalent_assets = []
    if "barrett" in robot_text:
        equivalent_assets.append((BARRETT_URDF, BARRETT_MJCF))
    if "allegro" in robot_text:
        equivalent_assets.append((ALLEGRO_RIGHT_URDF, ALLEGRO_RIGHT_MJCF))
    if "wujihand" in robot_text:
        equivalent_assets.append(
            (WUJIHAND_RIGHT_XML.with_suffix(".urdf"), WUJIHAND_RIGHT_XML)
        )

    return any(
        expected_path == Path(expected).expanduser().resolve()
        and actual_path == Path(actual).expanduser().resolve()
        for expected, actual in equivalent_assets
    )


def _ensure_source_cube_size(
    env: TabletopManipulationEnv,
    source: dict[str, np.ndarray],
    trajectory_index: int,
) -> None:
    if not hasattr(env.task, "_set_cube_size"):
        return
    side_length = _trajectory_value(
        source, "initial_cube_side_length", trajectory_index
    )
    if side_length is None:
        side_length = _trajectory_value(source, "cube_side_length", trajectory_index)
    height = _trajectory_value(source, "initial_cube_height", trajectory_index)
    if height is None:
        height = _trajectory_value(source, "cube_height", trajectory_index)
    if side_length is None:
        return

    desired = float(np.asarray(side_length).item())
    desired_height = None if height is None else float(np.asarray(height).item())
    obs = env.task.observe(env.model, env.data, env.scene)
    current = float(obs["cube_side_length"])
    current_height = float(obs.get("cube_height", current))
    if np.isclose(current, desired, atol=1e-7) and (
        desired_height is None or np.isclose(current_height, desired_height, atol=1e-7)
    ):
        return
    env.task._set_cube_size(
        env.model, env.scene, side_length=desired, height=desired_height
    )
    env.backend.forward()


def _initial_cube_pose(
    source: dict[str, np.ndarray], trajectory_index: int
) -> np.ndarray:
    if "initial_cube_pose" in source:
        pose = _indexed_pose(
            source["initial_cube_pose"], trajectory_index, key="initial_cube_pose"
        )
        return np.asarray(pose, dtype=np.float32).reshape(7)
    pose_key = "cube_pose" if "cube_pose" in source else "object_pose"
    if pose_key not in source:
        raise ValueError(
            "Replay source without trajectory_seed must contain an object pose"
        )
    cube_pose = _indexed_pose(source[pose_key], trajectory_index, key=pose_key)
    arr = np.asarray(cube_pose, dtype=np.float32)
    if arr.ndim == 2:
        return arr[0].reshape(7)
    return arr.reshape(7)


def _indexed_pose(value: np.ndarray, trajectory_index: int, *, key: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.dtype == object and arr.ndim == 1:
        if int(trajectory_index) >= int(arr.shape[0]):
            raise IndexError(
                f"{key} has {arr.shape[0]} entries; index {trajectory_index} is out of range"
            )
        return np.asarray(arr[int(trajectory_index)])
    if arr.ndim == 1:
        if int(trajectory_index) != 0:
            raise IndexError(f"{key} is unbatched, so only trajectory index 0 is valid")
        return arr
    if arr.ndim == 2:
        if arr.shape[-1] != 7:
            raise ValueError(f"{key} must have pose last dimension 7, got {arr.shape}")
        return arr[int(trajectory_index)]
    if arr.ndim >= 3:
        return arr[int(trajectory_index)]
    raise ValueError(f"{key} must contain pose data, got shape {arr.shape}")


def _extend_flat_trace(
    dst: dict[str, list[np.ndarray]], src: dict[str, list[np.ndarray]]
) -> None:
    for key, frames in src.items():
        if key.startswith("_"):
            continue
        if key not in dst:
            dst[key] = []
        dst[key].extend(frames)


def _should_report_progress(local_idx: int, total: int, progress_interval: int) -> bool:
    done = int(local_idx) + 1
    return done == int(total) or (
        int(progress_interval) > 0 and done % int(progress_interval) == 0
    )


def _print_progress(results: list[RolloutResult], total: int) -> None:
    done = len(results)
    successes = sum(1 for result in results if result.success)
    rate = successes / done if done else 0.0
    print(
        f"progress {done}/{total}: successes={successes} success_rate={rate:.3%}",
        flush=True,
    )


def _print_summary(results: list[RolloutResult]) -> None:
    successes = sum(1 for result in results if result.success)
    rate = successes / len(results) if results else 0.0
    failed_indices = [result.source_index for result in results if not result.success]
    error_indices = [result.source_index for result in results if result.error]
    heights = np.asarray([result.max_cube_z for result in results], dtype=np.float32)
    finite = heights[np.isfinite(heights)]

    print(
        f"total_trajectories={len(results)} successes={successes} "
        f"success_rate={rate:.3%}",
        flush=True,
    )
    print(f"failed_indices={failed_indices}", flush=True)
    if finite.size:
        print(
            "max_cube_z_summary="
            f"min={float(np.min(finite)):.4f} "
            f"mean={float(np.mean(finite)):.4f} "
            f"max={float(np.max(finite)):.4f}",
            flush=True,
        )
    else:
        print("max_cube_z_summary=min=nan mean=nan max=nan", flush=True)
    if error_indices:
        print(f"error_indices={error_indices}", flush=True)
        for result in results:
            if result.error:
                print(
                    f"trajectory {result.source_index} error: {result.error}",
                    flush=True,
                )


def _compact_payload(
    results: list[RolloutResult],
    *,
    cfg: DictConfig,
    env: TabletopManipulationEnv,
    source: dict[str, np.ndarray],
    source_path: Path,
    selected: list[int],
    source_count: int,
    backend: str,
    dt: float,
) -> dict[str, object]:
    source_seed = np.asarray(
        [
            -1 if result.source_seed is None else int(result.source_seed)
            for result in results
        ],
        dtype=np.int32,
    )
    has_source_seed = np.asarray(
        [result.source_seed is not None for result in results], dtype=np.bool_
    )
    success = np.asarray([result.success for result in results], dtype=np.bool_)
    max_cube_z = np.asarray([result.max_cube_z for result in results], dtype=np.float32)
    failed_indices = np.asarray(
        [result.source_index for result in results if not result.success],
        dtype=np.int32,
    )
    error_indices = np.asarray(
        [result.source_index for result in results if result.error], dtype=np.int32
    )
    error_messages = np.asarray([result.error for result in results])

    payload: dict[str, object] = {
        "schema_version": np.asarray(1, dtype=np.int32),
        "source_pt_path": np.asarray(str(source_path.expanduser().resolve())),
        "source_trajectory_count": np.asarray(int(source_count), dtype=np.int32),
        "source_indices": np.asarray(selected, dtype=np.int32),
        "embodiment": np.asarray(str(cfg.env.embodiment)),
        "mode": np.asarray(str(cfg.env.mode)),
        "backend": np.asarray(str(backend)),
        "dt": np.asarray(float(dt), dtype=np.float32),
        "frame_skip": np.asarray(int(env.backend.frame_skip), dtype=np.int32),
        "controller_steps_per_prediction": np.asarray(
            int(
                _positive_int_from_cfg(
                    cfg, "online_rollout.controller_steps_per_prediction", default=1
                )
            ),
            dtype=np.int32,
        ),
        "source_dt": np.asarray(
            float(_source_scalar(source, "dt") or np.nan), dtype=np.float32
        ),
        "effective_action_dt": np.asarray(
            float(dt)
            * int(
                _positive_int_from_cfg(
                    cfg, "online_rollout.controller_steps_per_prediction", default=1
                )
            ),
            dtype=np.float32,
        ),
        "gravity_compensation": np.asarray(env.gravity_compensation),
        "enable_fingertip_collision_pads": np.asarray(
            bool(
                OmegaConf.select(
                    cfg, "env.enable_fingertip_collision_pads", default=True
                )
            )
        ),
        "success": success,
        "success_count": np.asarray(int(np.sum(success)), dtype=np.int32),
        "success_rate": np.asarray(
            float(np.mean(success)) if success.size else 0.0, dtype=np.float32
        ),
        "failed_indices": failed_indices,
        "error_indices": error_indices,
        "error_messages": error_messages,
        "completed_steps": np.asarray(
            [result.completed_steps for result in results], dtype=np.int32
        ),
        "action_frame_count": np.asarray(
            [result.action_frame_count for result in results], dtype=np.int32
        ),
        "max_cube_z": max_cube_z,
        "final_cube_pose": np.asarray(
            [result.final_cube_pose for result in results], dtype=np.float32
        ),
        "source_trajectory_seed": source_seed,
        "has_source_trajectory_seed": has_source_seed,
        "robot_xml_path": np.asarray(str(env.scene.robot_xml_path)),
        "group_names": np.asarray(env.robot.group_names),
    }
    if hasattr(env.task, "lift_success_z"):
        payload["lift_success_z"] = np.asarray(
            float(env.task.lift_success_z), dtype=np.float32
        )
    actuator_control = getattr(env, "actuator_control", None)
    if actuator_control is not None:
        for key in (
            "arm_kp_scale",
            "arm_kv_scale",
            "arm_force_scale",
            "hand_kp_scale",
            "hand_kv_scale",
            "hand_force_scale",
        ):
            payload[f"controller_{key}"] = np.asarray(
                float(getattr(actuator_control, key)), dtype=np.float32
            )
        for group_name in ("base_xyz", "base_rpy", "fingers"):
            group = getattr(actuator_control, group_name, None)
            if group is None:
                continue
            for key in ("kp", "kv", "max_effort"):
                value = getattr(group, key)
                if value is not None:
                    payload[f"controller_{group_name}_{key}"] = _controller_value_array(
                        value
                    )
    for key in (
        "embodiment",
        "action_space_type",
        "mode",
        "frame_skip",
        "cube_x_range",
        "cube_y_range",
        "cube_side_length_range",
        "cube_height_ratio",
        "cube_height_range",
        "cube_yaw_range_deg",
    ):
        if key in source:
            payload[f"source_{key}"] = np.asarray(source[key])
    return payload


def _controller_value_array(value: object) -> np.ndarray:
    if isinstance(value, str):
        return np.asarray(value)
    return np.asarray(float(value), dtype=np.float32)


def _save_vis_trace_npz(out: str | Path, payload: dict[str, object]) -> Path:
    out_path = Path(out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, **payload)
    return out_path
