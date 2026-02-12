import sys, types

# Build a fake module "graspnetAPI.utils.eval_utils" with the exact names
# that graspnetAPI.graspnet_eval imports. None of these are used by the
# AnyGrasp *detection* demo, so they can be dummies.
m = types.ModuleType("graspnetAPI.utils.eval_utils")

def _not_available(*a, **k):
    raise RuntimeError("Dex-Net evaluation disabled: eval_utils stub in use")

for name in [
    "get_scene_name",
    "create_table_points",
    "parse_posevector",
    "load_dexnet_model",
    "transform_points",
    "compute_point_distance",
    "compute_closest_points",
    "voxel_sample_points",
    "topk_grasps",
    "get_grasp_score",
    "collision_detection",
    "eval_grasp",
]:
    setattr(m, name, _not_available)

# Register the stub *before* graspnetAPI is imported
sys.modules["graspnetAPI.utils.eval_utils"] = m

# Optional belt-and-suspenders: if anything tries to import dexnet/meshpy directly,
# make them empty modules so import succeeds (they won't be called by the demo).
sys.modules.setdefault("dexnet", types.ModuleType("dexnet"))
sys.modules.setdefault("dexnet.grasping", types.ModuleType("dexnet.grasping"))
sys.modules.setdefault("meshpy", types.ModuleType("meshpy"))
sys.modules.setdefault("meshpy.obj_file", types.ModuleType("meshpy.obj_file"))
