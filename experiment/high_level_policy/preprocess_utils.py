from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation as R

def pose_mats_to_pos_quat_wxyz(pose_mats: np.ndarray) -> np.ndarray:
    pose_np = np.asarray(pose_mats, dtype=np.float32)
    rot = R.from_matrix(pose_np[:, :3, :3].astype(np.float64))
    quat_xyzw = rot.as_quat().astype(np.float32)
    quat_wxyz = np.concatenate([quat_xyzw[:, 3:4], quat_xyzw[:, :3]], axis=-1)
    return np.concatenate([pose_np[:, :3, 3], quat_wxyz], axis=-1).astype(np.float32)

