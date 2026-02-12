#!/usr/bin/env python3
import os
import argparse
import logging
from time import time
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

# Evaluator (prefer official graspnetAPI)
from graspnetAPI import GraspNetEval
try:
    from graspnetAPI import GraspGroup
except Exception:
    GraspGroup = None

# AnyGrasp
try:
    from gsnet import AnyGrasp
except Exception as e:
    AnyGrasp = None
    _ANYGRASP_IMPORT_ERROR = e


# ---------------------------
# Logging / perf controls
# ---------------------------
def setup_logger():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s", force=True)


def hard_cap_threads(n: int):
    os.environ["OMP_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["NUMEXPR_NUM_THREADS"] = str(n)
    try:
        torch.set_num_threads(n)
        torch.set_num_interop_threads(1)
    except Exception:
        pass


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------
# Dataset IO helpers
# ---------------------------
def scene_cam_dir(graspnet_root: str, scene_id: int, camera: str) -> str:
    return os.path.join(graspnet_root, "scenes", f"scene_{scene_id:04d}", camera)


def load_camK(graspnet_root: str, scene_id: int, camera: str) -> np.ndarray:
    p = os.path.join(scene_cam_dir(graspnet_root, scene_id, camera), "camK.npy")
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing camK.npy: {p}")
    return np.load(p).astype(np.float32).reshape(3, 3)


def load_depth_scale(graspnet_root: str, scene_id: int, camera: str, default_scale: float) -> float:
    p = os.path.join(scene_cam_dir(graspnet_root, scene_id, camera), "depth_scale.npy")
    if os.path.exists(p):
        v = np.load(p)
        v = float(np.array(v).reshape(-1)[0])
        if v > 0:
            return v
    return float(default_scale)


def load_rgb_depth(graspnet_root: str, scene_id: int, camera: str, ann_id: int):
    base = scene_cam_dir(graspnet_root, scene_id, camera)
    rgb_p = os.path.join(base, "rgb", f"{ann_id:04d}.png")
    dep_p = os.path.join(base, "depth", f"{ann_id:04d}.png")

    if not os.path.exists(rgb_p):
        rgb_p = os.path.join(base, "rgb", f"{ann_id:04d}.jpg")

    if not os.path.exists(rgb_p):
        raise FileNotFoundError(f"Missing RGB: {rgb_p}")
    if not os.path.exists(dep_p):
        raise FileNotFoundError(f"Missing depth: {dep_p}")

    rgb = np.array(Image.open(rgb_p).convert("RGB"), dtype=np.uint8)
    depth = np.array(Image.open(dep_p), dtype=np.uint16)
    return rgb, depth


# ---------------------------
# Point cloud + workspace
# ---------------------------
def rgbd_to_points(rgb: np.ndarray, depth_u16: np.ndarray, K: np.ndarray,
                   depth_scale: float,
                   z_min: float, z_max: float,
                   max_points: int):
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])

    z = depth_u16.astype(np.float32) / float(depth_scale)
    mask = (z > float(z_min)) & (z < float(z_max))
    v, u = np.where(mask)
    if v.size == 0:
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.float32)

    zz = z[v, u]
    xx = (u.astype(np.float32) - cx) * zz / fx
    yy = (v.astype(np.float32) - cy) * zz / fy
    pts = np.stack([xx, yy, zz], axis=1).astype(np.float32)
    cols = (rgb[v, u].astype(np.float32) / 255.0).astype(np.float32)

    if max_points > 0 and pts.shape[0] > max_points:
        # deterministic-ish: shuffle by permutation
        idx = np.random.choice(pts.shape[0], max_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]
    return pts, cols


def auto_workspace_lims(pts: np.ndarray, margin=(0.05, 0.05, 0.05)):
    """
    Robust percentile-based cropping.
    Better than min/max (min/max explodes on outliers).
    """
    if pts.shape[0] == 0:
        return [-0.6, 0.6, -0.6, 0.6, 0.02, 1.5]

    lo = np.percentile(pts, 1, axis=0)
    hi = np.percentile(pts, 99, axis=0)
    mx, my, mz = margin

    xmin, ymin, zmin = lo[0] - mx, lo[1] - my, lo[2] - mz
    xmax, ymax, zmax = hi[0] + mx, hi[1] + my, hi[2] + mz

    xmin, xmax = float(max(xmin, -1.0)), float(min(xmax, 1.0))
    ymin, ymax = float(max(ymin, -1.0)), float(min(ymax, 1.0))
    zmin, zmax = float(max(zmin, 0.0)), float(min(zmax, 2.0))
    return [xmin, xmax, ymin, ymax, zmin, zmax]


# ---------------------------
# AnyGrasp init + conversion
# ---------------------------
def init_anygrasp(checkpoint_path: str, max_gripper_width: float, gripper_height: float,
                 top_down: bool, debug: bool):
    if AnyGrasp is None:
        raise RuntimeError(f"Failed to import AnyGrasp:\n{_ANYGRASP_IMPORT_ERROR}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # clamp like common AnyGrasp demos
    mgw = max(0.0, min(0.12, float(max_gripper_width)))

    cfg = SimpleNamespace(
        checkpoint_path=checkpoint_path,
        max_gripper_width=mgw,
        gripper_height=float(gripper_height),
        top_down_grasp=bool(top_down),
        debug=bool(debug),
    )
    m = AnyGrasp(cfg)
    m.load_net()
    return m, cfg


def gg_to_nx17(gg):
    """
    Convert AnyGrasp GraspGroup-like output into GraspNet Nx17:
    [score, width, height, depth, R(9), t(3), object_id]
    """
    # Some versions may already expose numpy conversion
    for attr in ["to_numpy", "as_numpy", "to_array"]:
        if hasattr(gg, attr):
            a = np.asarray(getattr(gg, attr)())
            if a.ndim == 2 and a.shape[1] == 17:
                return a.astype(np.float32)

    rows = []
    for g in gg:
        score = float(getattr(g, "score"))
        width = float(getattr(g, "width"))
        height = float(getattr(g, "height", 0.03))
        depth = float(getattr(g, "depth", 0.02))
        R = np.asarray(getattr(g, "rotation_matrix"), dtype=np.float32).reshape(3, 3)
        t = np.asarray(getattr(g, "translation"), dtype=np.float32).reshape(3,)
        obj_id = float(getattr(g, "object_id", -1))
        row = np.concatenate([
            np.array([score, width, height, depth], np.float32),
            R.reshape(-1).astype(np.float32),
            t.astype(np.float32),
            np.array([obj_id], np.float32),
        ])
        rows.append(row)

    if not rows:
        return np.zeros((0, 17), np.float32)
    return np.stack(rows, axis=0).astype(np.float32)


def postprocess_gg(gg):
    """
    Apply NMS (if available) + sort_by_score (if available).
    Keeps output benchmark-safe: no padding, no collision-off filling.
    """
    if gg is None:
        return gg

    # optional NMS (some AnyGrasp forks provide it)
    if hasattr(gg, "nms"):
        try:
            gg = gg.nms()
        except Exception:
            pass

    # sort
    if hasattr(gg, "sort_by_score"):
        try:
            gg = gg.sort_by_score()
        except Exception:
            pass
    else:
        # python-side sort fallback
        try:
            gg = sorted(list(gg), key=lambda g: float(getattr(g, "score", 0.0)), reverse=True)
        except Exception:
            pass

    return gg


def save_npy(path: str, arr_nx17: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if GraspGroup is not None:
        GraspGroup(arr_nx17).save_npy(path)
    else:
        np.save(path, arr_nx17)


# ---------------------------
# Eval metrics helper
# ---------------------------
def compute_ap_from_res(res: np.ndarray):
    # res: (S,256,50,6)
    aps = res.mean(axis=(0, 1, 2))  # (6,)
    ap = float(aps.mean())
    ap04 = float(aps[1])
    ap08 = float(aps[3])
    return aps, ap, ap04, ap08


# ---------------------------
# Main modes
# ---------------------------
def inference(args):
    logging.info("=== Inference (dump) ===")
    m, cfg = init_anygrasp(args.checkpoint_path, args.max_gripper_width, args.gripper_height,
                           args.top_down, args.debug)

    for sid in range(args.scene_l, args.scene_r):
        logging.info(f"Scene {sid:04d}")
        K = load_camK(args.graspnet_root, sid, args.camera)
        depth_scale = load_depth_scale(args.graspnet_root, sid, args.camera, args.depth_scale)
        logging.info(f"  depth_scale={depth_scale}")

        for ann_id in range(args.start_frame, args.end_frame):
            out_path = os.path.join(args.dump_dir, f"scene_{sid:04d}", args.camera, f"{ann_id:04d}.npy")
            if args.skip_existing and os.path.exists(out_path):
                continue

            rgb, depth = load_rgb_depth(args.graspnet_root, sid, args.camera, ann_id)
            pts, cols = rgbd_to_points(
                rgb, depth, K,
                depth_scale=depth_scale,
                z_min=args.z_min,
                z_max=args.z_max,
                max_points=args.max_points,
            )

            if pts.shape[0] == 0:
                save_npy(out_path, np.zeros((0, 17), np.float32))
                continue

            lims = args.lims if args.lims is not None else auto_workspace_lims(pts)

            t0 = time()
            gg, _ = m.get_grasp(
                pts, cols,
                lims=lims,
                apply_object_mask=bool(args.apply_object_mask),
                dense_grasp=bool(args.dense_grasp),
                collision_detection=bool(args.collision_detection),
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            dt_ms = (time() - t0) * 1000.0

            gg = postprocess_gg(gg)

            if gg is None or len(gg) == 0:
                arr = np.zeros((0, 17), np.float32)
            else:
                gg = gg[:min(args.topk_save, 50)]
                arr = gg_to_nx17(gg)

            save_npy(out_path, arr)

            if ann_id % args.log_every == 0:
                top_score = float(arr[0, 0]) if arr.shape[0] > 0 else -1.0
                logging.info(f"  frame {ann_id:04d} grasps={arr.shape[0]:2d} top_score={top_score:.4f} pts={pts.shape[0]:5d} {dt_ms:.1f} ms")

    logging.info("Inference done.")


def evaluate(args):
    logging.info("=== Evaluation ===")
    ge = GraspNetEval(root=args.graspnet_root, camera=args.camera, split=args.split)

    all_scene_acc = []
    for sid in range(args.scene_l, args.scene_r):
        logging.info(f"eval_scene {sid:04d}")
        acc = ge.eval_scene(
            scene_id=sid,
            dump_folder=args.dump_dir,
            TOP_K=50,
            vis=False,
            max_width=args.max_gripper_width,
        )
        all_scene_acc.append(np.asarray(acc, dtype=np.float32))

    res = np.stack(all_scene_acc, axis=0)  # (S,256,50,6)
    np.save(args.save, res)

    aps_vec, ap, ap04, ap08 = compute_ap_from_res(res)
    logging.info(f"Saved: {args.save} shape={res.shape}")
    logging.info(f"AP@mu [0.2,0.4,0.6,0.8,1.0,1.2]: {aps_vec.tolist()}")
    logging.info(f"AP(mean over mu): {ap:.12f}")
    logging.info(f"AP@0.4: {ap04:.12f}")
    logging.info(f"AP@0.8: {ap08:.12f}")


def main():
    setup_logger()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["infer", "eval", "both"], default="both")

    ap.add_argument("--graspnet-root", required=True)
    ap.add_argument("--camera", choices=["realsense", "kinect"], default="realsense")

    # GraspNetEval split initialization (keep test unless you know your API variant expects something else)
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])

    ap.add_argument("--scene-l", type=int, default=160)
    ap.add_argument("--scene-r", type=int, default=161)

    ap.add_argument("--checkpoint-path", required=True)
    ap.add_argument("--dump-dir", required=True)
    ap.add_argument("--save", default="temp_result.npy")

    ap.add_argument("--cpu-threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)

    # point cloud (ideal-for-8GB defaults)
    ap.add_argument("--max-points", type=int, default=30000)
    ap.add_argument("--z-min", type=float, default=0.02)
    ap.add_argument("--z-max", type=float, default=1.50)
    ap.add_argument("--depth-scale", type=float, default=1000.0)

    # anygrasp options (ideal)
    ap.add_argument("--max-gripper-width", type=float, default=0.10)
    ap.add_argument("--gripper-height", type=float, default=0.03)
    ap.add_argument("--top-down", action="store_true")
    ap.add_argument("--debug", action="store_true")

    ap.add_argument("--apply-object-mask", action="store_true", default=True)
    ap.add_argument("--dense-grasp", action="store_true", default=False)
    ap.add_argument("--collision-detection", action="store_true", default=True)

    # workspace: if not provided, auto workspace is used
    ap.add_argument("--lims", type=float, nargs=6, default=None,
                    help="xmin xmax ymin ymax zmin zmax (meters). If omitted, auto workspace is used.")

    ap.add_argument("--topk-save", type=int, default=50)
    ap.add_argument("--skip-existing", action="store_true", default=False)

    ap.add_argument("--start-frame", type=int, default=0)
    ap.add_argument("--end-frame", type=int, default=256)
    ap.add_argument("--log-every", type=int, default=25)

    args = ap.parse_args()

    hard_cap_threads(args.cpu_threads)
    set_seed(args.seed)

    logging.info(f"mode={args.mode} scenes=[{args.scene_l},{args.scene_r}) camera={args.camera} split={args.split}")
    logging.info(f"dump_dir={args.dump_dir} save={args.save}")
    logging.info(f"max_points={args.max_points} depth_scale_default={args.depth_scale} z=[{args.z_min},{args.z_max}] topk_save={args.topk_save}")
    logging.info(f"mask={args.apply_object_mask} dense={args.dense_grasp} collision={args.collision_detection} lims={args.lims}")

    if args.mode in ("infer", "both"):
        inference(args)
    if args.mode in ("eval", "both"):
        evaluate(args)


if __name__ == "__main__":
    main()
