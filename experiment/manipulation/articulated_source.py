"""Original commanded hand FK and measured objects for articulated demonstrations.

Geometry is sampled once, independently of trajectories or target embodiments.
The paper selector in sparse_contact_extraction owns all contact selection.
"""
from functools import lru_cache

import mujoco
import numpy as np


def _descends(model, body, root):
    while body:
        if body == root:
            return True
        body = int(model.body_parentid[body])
    return False


def _sample_geoms(model, ids, count, seed, *, convex=False):
    from experiment.encode_contacts import _allocate_counts, _geom_local_mesh, _sample_mesh_surface

    meshes = [(int(i), _geom_local_mesh(model, int(i))) for i in ids]
    meshes = [(i, m) for i, m in meshes if m is not None and m.area > 0]
    if convex:
        meshes = [(i, m.convex_hull) for i, m in meshes]
    if not meshes:
        raise ValueError("No source geometry available for surface sampling")
    counts = _allocate_counts(np.array([m.area for _, m in meshes]), count)
    points, normals, owners = [], [], []
    for (geom, mesh), n in zip(meshes, counts):
        p, normal = _sample_mesh_surface(mesh, int(n), seed=seed + geom)
        points.append(p)
        normals.append(normal)
        owners.extend([geom] * len(p))
    return np.concatenate(points), np.concatenate(normals), np.array(owners, dtype=np.int32)


def _world(data, local, normals, owners):
    rotations = data.geom_xmat[owners].reshape(-1, 3, 3)
    return (
        np.einsum("nij,nj->ni", rotations, local) + data.geom_xpos[owners],
        np.einsum("nij,nj->ni", rotations, normals),
    )


@lru_cache(maxsize=1)
def _source_hand_model():
    import torch
    from robot_model.robot_model import create_robot_model
    return create_robot_model('wujihand_right', device=torch.device('cpu'),
                              num_points=128, load_surface_assets=False)


def commanded_hand_trajectory(episode, model, local, normal, owners):
    """Use the exact source command fields used by the original task encoder."""
    import torch
    from scipy.spatial.transform import Rotation
    from experiment.encode_contacts import _episode_source_array
    fields = [_episode_source_array(episode, key, min_tail_ndim=1)
              for key in ('action_ee_pos', 'action_ee_quat_wxyz', 'action_hand_q')]
    if any(value is None for value in fields):
        raise ValueError('Articulated demonstrations require their original recorded hand command fields')
    xyz, quat, joints = fields
    q = np.concatenate([xyz, Rotation.from_quat(quat[:, [1, 2, 3, 0]]).as_euler('XYZ'), joints], axis=1).astype(np.float32)
    hand = _source_hand_model()
    if q.shape[1] != hand.dof:
        raise ValueError(f'Original hand commands must have {hand.dof} entries')
    geom_rotations = Rotation.from_quat(model.geom_quat[owners][:, [1, 2, 3, 0]]).as_matrix()
    body_local = np.einsum('nij,nj->ni', geom_rotations, local) + model.geom_pos[owners]
    body_normal = np.einsum('nij,nj->ni', geom_rotations, normal)
    bodies = model.geom_bodyid[owners]
    points = np.empty((len(q), len(local), 3), np.float32)
    normals = np.empty_like(points)
    with torch.inference_mode():
        hand.update_status(torch.from_numpy(q))
        for body in np.unique(bodies):
            ids = np.flatnonzero(bodies == body)
            body_name = model.body(int(body)).name
            if body_name == 'right_wuji_palm':
                body_name = 'right_palm_link'
            tf = hand.frame_status[body_name].get_matrix().numpy()
            points[:, ids] = np.einsum('tij,nj->tni', tf[:, :3, :3], body_local[ids]) + tf[:, None, :3, 3]
            normals[:, ids] = np.einsum('tij,nj->tni', tf[:, :3, :3], body_normal[ids])
    return q, points, normals


@lru_cache(maxsize=8)
def source_template(task, object_id, count, object_count=4096):
    from experiment.encode_contacts import (
        CROSS_DIR, _exposed_template_sample_mask, _exterior_surface_neighbors,
        _short_hash_from_arrays,
    )
    from experiment.tabletop_sim.scene import compile_tabletop_scene

    kwargs = {}
    if task == "pumpbottle_use":
        kwargs = dict(pump_bottle_asset_id=object_id,
                      pump_bottle_manifest_path=str(CROSS_DIR / "experiment/assets/pump_bottle_suite.json"))
    elif task != "keyboard_press":
        raise ValueError(f"Unsupported articulated source task: {task}")
    _, model, data, scene = compile_tabletop_scene(
        embodiment="marvin_wuji", add_cameras=False,
        object_kind="pump_bottle" if task == "pumpbottle_use" else "mini_keyboard",
        **kwargs,
    )
    data.qpos[:] = model.qpos0
    mujoco.mj_forward(model, data)
    hand_root = model.body('right_wuji_palm').id
    hand_geoms = [i for i in range(model.ngeom)
                  if _descends(model, int(model.geom_bodyid[i]), hand_root)
                  and (model.geom_contype[i] or model.geom_conaffinity[i])]
    # MuJoCo's mesh contact geometry is its convex hull. Sampling that surface
    # excludes cavities/triangles that are internal to the physical collider.
    # `count` is the final template size, after removing internal samples.
    count = int(count)
    if count < 2:
        raise ValueError('A source surface template requires at least two points')
    for multiplier in (2, 4, 8, 16):
        local, normal, owners = _sample_geoms(
            model, hand_geoms, multiplier * count, seed=7, convex=True)
        canonical, canonical_normal = _world(data, local, normal, owners)
        exposed = _exposed_template_sample_mask(model, data, canonical, canonical_normal, owners)
        pool = [x[exposed] for x in (local, normal, owners, canonical, canonical_normal)]
        if len(pool[0]) >= count:
            indices = np.sort(np.random.default_rng(7).choice(len(pool[0]), count, replace=False))
            local, normal, owners, canonical, canonical_normal = [x[indices].copy() for x in pool]
            break
    else:
        raise ValueError(f'Cannot sample {count} exposed source-surface points')
    # Contact patches can span adjacent physical pieces of one finger. Do not
    # split a connected patch at a MuJoCo body boundary or at a sharp normal.
    from scipy.spatial import cKDTree
    neighbors = [[] for _ in canonical]
    for i, j in cKDTree(canonical).query_pairs(.01):
        neighbors[i].append(j)
        neighbors[j].append(i)
    neighbors = tuple(tuple(row) for row in neighbors)
    edges = np.array([(i, j) for i, row in enumerate(neighbors) for j in row], dtype=np.int64).reshape(-1, 2)
    # Use body names instead of scene-dependent geometry IDs in the identity.
    body_names = np.array([model.body(model.geom_bodyid[i]).name for i in owners])
    template_hash = _short_hash_from_arrays(local, normal, body_names, canonical, prefix="wuji_collision_surface_fixed_count_v4")
    graph_hash = _short_hash_from_arrays(edges, prefix="wuji_collision_surface_fixed_count_graph_v5")
    root = model.body(scene.mini_keyboard_body_name if task == "keyboard_press"
                      else scene.pump_bottle_body_name).id
    object_geoms = [i for i in range(model.ngeom)
                    if _descends(model, int(model.geom_bodyid[i]), root)
                    and (model.geom_contype[i] or model.geom_conaffinity[i])]
    object_surface = _sample_geoms(model, object_geoms, int(object_count), seed=42, convex=True)
    return model, data, local, normal, owners, canonical, neighbors, template_hash, graph_hash, object_surface


def surface_trajectory(episode, count, object_count=4096, source_configuration='commanded'):
    from experiment.encode_contacts import SurfaceTrajectory

    task = episode["source_meta"]["task"]
    object_id = episode["source_episode_meta"].get("object_id", "pump_bottle005")
    model, data, local, normal, owners, canonical, neighbors, template_hash, graph_hash, obj = source_template(task, object_id, count, object_count)
    q = np.asarray(episode["manipulator_q_source"], dtype=np.float64)
    if q.ndim != 2 or q.shape[1] != model.nq or not np.isfinite(q).all():
        raise ValueError(f"Measured {task} qpos must have shape (T, {model.nq}); got {q.shape}")
    if source_configuration not in {'commanded', 'measured'}:
        raise ValueError('source_configuration must be commanded or measured')
    if source_configuration == 'commanded':
        commands, points, normals = commanded_hand_trajectory(episode, model, local, normal, owners)
    else:
        from scipy.spatial.transform import Rotation
        commands = np.empty((len(q), 26), dtype=np.float32)
        points = np.empty((len(q), len(local), 3), dtype=np.float32)
        normals = np.empty_like(points)
        palm = model.body('right_wuji_palm').id
        joint_addresses = [int(model.joint(name).qposadr[0])
                           for name in _source_hand_model().get_joint_orders()[6:]]
    object_points = np.empty((len(q), len(obj[0]), 3), dtype=np.float32)
    object_normals = np.empty_like(object_points)
    for t, configuration in enumerate(q):
        data.qpos[:] = configuration
        if task == "keyboard_press":
            # Keyboard placement is a fixed body transform, not a free joint.
            body = model.body("mini_keyboard").id
            model.body_pos[body] = episode["object_pose"][t, :3]
            model.body_quat[body] = episode["object_pose"][t, 3:]
        mujoco.mj_forward(model, data)
        if source_configuration == 'measured':
            points[t], normals[t] = _world(data, local, normal, owners)
            commands[t, :3] = data.xpos[palm]
            commands[t, 3:6] = Rotation.from_matrix(data.xmat[palm].reshape(3, 3)).as_euler('XYZ')
            commands[t, 6:] = data.qpos[joint_addresses]
        object_points[t], object_normals[t] = _world(data, *obj)
    normals /= np.maximum(np.linalg.norm(normals, axis=-1, keepdims=True), 1e-8)
    episode["contact_scoring_points_world"] = object_points
    episode["contact_scoring_normals_world"] = object_normals
    return SurfaceTrajectory(points, normals, canonical, template_hash, len(local), commands,
                             f"{source_configuration}_articulated_hand_fk", "wujihand_right", neighbors, graph_hash)
