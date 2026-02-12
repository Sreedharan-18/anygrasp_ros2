#!/usr/bin/env python3

import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from gsnet import AnyGrasp
from graspnetAPI.grasp import GraspGroup


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion (x,y,z,w)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        if (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s

    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    # normalize
    n = np.linalg.norm(q)
    if n > 0:
        q = q / n
    return q.astype(np.float32)


def atomic_write_json(path: Path, data: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)  # atomic on POSIX


def run_anygrasp_and_save(cfgs):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    data_dir = Path(cfgs.data_dir).expanduser().resolve()
    color_path = data_dir / "ros2_color.png"
    depth_path = data_dir / "ros2_depth.png"

    if not color_path.exists():
        raise FileNotFoundError(f"Missing: {color_path}")
    if not depth_path.exists():
        raise FileNotFoundError(f"Missing: {depth_path}")

    colors = np.array(Image.open(color_path), dtype=np.float32) / 255.0
    depths = np.array(Image.open(depth_path))  # uint16 mm

    # intrinsics
    fx, fy = cfgs.fx, cfgs.fy
    cx, cy = cfgs.cx, cfgs.cy
    scale = cfgs.depth_scale  # 1000 for mm->m

    # workspace limits for filtering grasps
    lims = [cfgs.xmin, cfgs.xmax, cfgs.ymin, cfgs.ymax, cfgs.zmin, cfgs.zmax]

    # depth -> point cloud (camera frame)
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)

    points_z = depths.astype(np.float32) / float(scale)  # meters
    points_x = (xmap.astype(np.float32) - cx) / fx * points_z
    points_y = (ymap.astype(np.float32) - cy) / fy * points_z

    # basic mask (you can tune)
    mask = (points_z > 0.0) & (points_z < cfgs.max_depth_m)
    points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
    colors_f = colors[mask].astype(np.float32)

    gg, cloud = anygrasp.get_grasp(
        points, colors_f, lims=lims,
        apply_object_mask=True,
        dense_grasp=False,
        collision_detection=True
    )

    out_json = Path(cfgs.output_json).expanduser().resolve()

    if len(gg) == 0:
        payload = {
            "ok": False,
            "reason": "NO_GRASP",
            "timestamp_unix": time.time(),
            "frame_id": cfgs.frame_id,
        }
        atomic_write_json(out_json, payload)
        print(f"[anygrasp] No grasp detected. Wrote: {out_json}")
        return 1

    gg = gg.nms().sort_by_score()
    best = gg[0]

    center = np.asarray(best.translation, dtype=np.float32).reshape(3)
    R = np.asarray(best.rotation_matrix, dtype=np.float32).reshape(3, 3)
    quat = rotmat_to_quat(R)

    payload = {
        "ok": True,
        "timestamp_unix": time.time(),
        "frame_id": cfgs.frame_id,  # IMPORTANT: this is the frame your points are in (camera frame)
        "position": {"x": float(center[0]), "y": float(center[1]), "z": float(center[2])},
        "orientation": {"x": float(quat[0]), "y": float(quat[1]), "z": float(quat[2]), "w": float(quat[3])},
        "width": float(best.width),
        "score": float(best.score),
        "lims": {"xmin": cfgs.xmin, "xmax": cfgs.xmax, "ymin": cfgs.ymin, "ymax": cfgs.ymax, "zmin": cfgs.zmin, "zmax": cfgs.zmax},
        "intrinsics": {"fx": fx, "fy": fy, "cx": cx, "cy": cy, "depth_scale": scale},
    }

    atomic_write_json(out_json, payload)
    print(f"[anygrasp] Wrote best grasp JSON: {out_json}")
    print(f"[anygrasp] score={payload['score']:.4f} width={payload['width']:.4f} frame={payload['frame_id']}")

    if cfgs.debug:
        trans_mat = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0],
                              [0, 0,-1, 0],
                              [0, 0, 0, 1]], dtype=np.float32)
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        for gripper in grippers:
            gripper.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, cloud])

    return 0


def main():
    parser = argparse.ArgumentParser()

    # AnyGrasp cfgs (what AnyGrasp expects)
    parser.add_argument("--checkpoint_path", required=True, help="Model checkpoint path")
    parser.add_argument("--max_gripper_width", type=float, default=0.8, help="Maximum gripper width")
    parser.add_argument("--gripper_height", type=float, default=0.03, help="Gripper height")
    parser.add_argument("--top_down_grasp", action="store_true", help="Output top-down grasps.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # Inputs/outputs
    parser.add_argument("--data_dir", default="/home/iki/ws_ros2/src/anygrasp_ros2/anygrasp_sdk/grasp_detection/example_data",
                        help="Directory containing ros2_color.png and ros2_depth.png")
    parser.add_argument("--output_json", default="/home/iki/ws_ros2/src/anygrasp_ros2/anygrasp_sdk/grasp_detection/example_data/best_grasp.json",
                        help="Where to write JSON (shared with ROS2 side)")
    parser.add_argument("--frame_id", default="color_optical_frame",
                        help="Frame_id for the grasp pose (camera frame in your reconstruction)")

    # Intrinsics (defaults = your current hardcoded values)
    parser.add_argument("--fx", type=float, default=927.17)
    parser.add_argument("--fy", type=float, default=927.37)
    parser.add_argument("--cx", type=float, default=651.32)
    parser.add_argument("--cy", type=float, default=349.62)
    parser.add_argument("--depth_scale", type=float, default=1000.0, help="Depth scale: 1000 if depth PNG is in mm")
    parser.add_argument("--max_depth_m", type=float, default=1.0)

    # Workspace lims
    parser.add_argument("--xmin", type=float, default=-0.19)
    parser.add_argument("--xmax", type=float, default=0.12)
    parser.add_argument("--ymin", type=float, default=0.02)
    parser.add_argument("--ymax", type=float, default=0.15)
    parser.add_argument("--zmin", type=float, default=0.0)
    parser.add_argument("--zmax", type=float, default=1.0)

    cfgs = parser.parse_args()

    # clamp like you did
    cfgs.max_gripper_width = max(0.0, min(0.12, cfgs.max_gripper_width))

    raise SystemExit(run_anygrasp_and_save(cfgs))


if __name__ == "__main__":
    main()
