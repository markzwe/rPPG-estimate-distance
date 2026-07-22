"""
utils/fps_points.py

Selects 468 canonical points from the BFM dense mesh using
Farthest Point Sampling (FPS), biased toward the five regions
of interest (right eye, left eye, lips, right cheek, left cheek).

Run this ONCE to generate the index file:
    python -c "from utils.fps_points import build_and_save_point_index; build_and_save_point_index()"

This creates two files:
    bfm_468_indices.json     — the 468 BFM vertex indices (load this every run)
    bfm_468_pointmap.json    — same indices + their 3D mean-face coordinates,
                               for sharing with InsightFace or other method users

How the sharing works:
    A user of another method (e.g. InsightFace) loads bfm_468_pointmap.json,
    which contains the 3D XYZ position of each of the 468 points on the BFM
    average face. They then find the nearest point in their own mesh to each
    of those 468 coordinates. This gives them the same anatomical locations,
    even though their tool has different internal point numbering.
    It is not pixel-perfect, but it is the best achievable correspondence
    across different face models.

Region budget (how the 468 slots are allocated):
    right_eye   : 40 points
    left_eye    : 40 points
    lips        : 60 points
    right_cheek : 30 points
    left_cheek  : 30 points
    global fill : 268 points  (FPS across the rest of the mesh)
    ─────────────────────────
    total       : 468 points

The 68-point sparse landmark indices (into the dense mesh) used as region
seeds come from the BFM-to-68-landmark correspondence that 3DDFA-V2 ships
with. Region seeds are the 2D sparse landmark groups projected into the
dense mesh via nearest-neighbour lookup on the mean face.
"""

import json
import os
import pickle
import numpy as np


# ---------------------------------------------------------------------------
# File paths (relative to 3DDFA_V2 repo root)
# ---------------------------------------------------------------------------
BFM_PATH        = "configs/bfm_noneck_v3.pkl"
INDEX_OUT       = "bfm_468_indices.json"
POINTMAP_OUT    = "bfm_468_pointmap.json"

# ---------------------------------------------------------------------------
# How many points to allocate to each region and the global fill
# ---------------------------------------------------------------------------
REGION_BUDGET = {
    "right_eye":   40,
    "left_eye":    40,
    "lips":        60,
    "right_cheek": 30,
    "left_cheek":  30,
}
TOTAL_POINTS = 468
GLOBAL_BUDGET = TOTAL_POINTS - sum(REGION_BUDGET.values())  # 268

# ---------------------------------------------------------------------------
# 68-point sparse landmark groups (indices into the 68-pt set, 0-based)
# used to seed each region in the dense mesh
# ---------------------------------------------------------------------------
SPARSE_REGION_SEEDS = {
    "right_eye":   list(range(36, 42)),   # 6 points
    "left_eye":    list(range(42, 48)),   # 6 points
    "lips":        list(range(48, 68)),   # 20 points
    "right_cheek": [2, 3, 4],            # jaw approximation
    "left_cheek":  [12, 13, 14],         # jaw approximation
}

# Nose tip index in the 68-pt set — used for nose-subtraction reference
NOSE_INDEX_68 = 30


# ---------------------------------------------------------------------------
# Core FPS algorithm
# ---------------------------------------------------------------------------

def farthest_point_sampling(points, n):
    """
    Select n points from a (N, 3) array using Farthest Point Sampling.

    Starts from a random point, then greedily picks the point that is
    farthest from all already-selected points.

    Args:
        points: np.ndarray shape (N, 3)
        n:      int, number of points to select (must be <= N)

    Returns:
        np.ndarray of int, shape (n,) — indices into `points`
    """
    N = len(points)
    if n >= N:
        return np.arange(N)

    selected = np.zeros(n, dtype=np.int64)
    distances = np.full(N, np.inf)

    # Start from a random point
    selected[0] = np.random.randint(0, N)
    for i in range(1, n):
        last = points[selected[i - 1]]
        # Distance from every point to the most recently added point
        dist_to_last = np.sum((points - last) ** 2, axis=1)
        # Keep the minimum distance to any already-selected point
        distances = np.minimum(distances, dist_to_last)
        # Pick the point with the largest minimum distance
        selected[i] = np.argmax(distances)

    return selected


# ---------------------------------------------------------------------------
# BFM loading
# ---------------------------------------------------------------------------

def load_bfm_mean_face():
    """
    Load the BFM mean face vertices from bfm/bfm.pkl.

    Returns:
        vertices: np.ndarray shape (38365, 3) — XYZ of mean face
        kpt_ind:  np.ndarray shape (68,)       — dense indices of 68 sparse landmarks
    """
    if not os.path.exists(BFM_PATH):
        raise FileNotFoundError(
            f"BFM file not found at {BFM_PATH}. "
            "Make sure you are running from the 3DDFA_V2 repo root."
        )

    with open(BFM_PATH, "rb") as f:
        bfm = pickle.load(f)

    # The mean shape is stored as a flat vector (x0,y0,z0, x1,y1,z1, ...)
    # Try common key names used across 3DDFA versions
    for key in ("u", "u_base", "mean_shape", "shapeMU"):
        if key in bfm:
            u = np.array(bfm[key]).flatten()
            break
    else:
        raise KeyError(
            f"Could not find mean shape in BFM. Keys available: {list(bfm.keys())}"
        )

    n_vertices = u.size // 3
    vertices = u.reshape(n_vertices, 3)

    # kpt_ind maps sparse landmarks → dense mesh vertex indices
    for key in ("kpt_ind", "keypoints", "kpt"):
        if key in bfm:
            kpt_ind = np.array(bfm[key], dtype=np.int64)
            break
    else:
        raise KeyError(
            f"Could not find keypoint indices in BFM. Keys: {list(bfm.keys())}"
        )

    # Filter out any keypoint indices that are out of bounds
    # (some BFM versions have invalid indices in the keypoints array)
    valid_mask = kpt_ind < n_vertices
    kpt_ind = kpt_ind[valid_mask]
    
    print(f"  Loaded {len(kpt_ind)} valid keypoints (filtered from {len(bfm[key])})")

    return vertices, kpt_ind


# ---------------------------------------------------------------------------
# Region-biased point selection
# ---------------------------------------------------------------------------

def _find_region_vertices(vertices, kpt_ind, sparse_indices, radius_fraction=0.12):
    """
    Find all dense mesh vertices within a radius of the given sparse landmark seeds.

    radius_fraction: fraction of the face bounding-box diagonal to use as radius.
    Larger = more vertices included in the region.
    """
    # Get seed positions in 3D
    seed_positions = vertices[kpt_ind[sparse_indices]]  # (K, 3)

    # Compute a radius based on face size
    bbox_diag = np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0))
    radius = radius_fraction * bbox_diag

    # Find all dense vertices within radius of any seed
    region_mask = np.zeros(len(vertices), dtype=bool)
    for seed in seed_positions:
        dist = np.linalg.norm(vertices - seed, axis=1)
        region_mask |= (dist < radius)

    return np.where(region_mask)[0]


def select_468_points(vertices, kpt_ind):
    """
    Select 468 points from the dense BFM mesh using region-biased FPS.

    Strategy:
      1. For each region, find all dense vertices near the sparse seed landmarks.
      2. Run FPS within each region to fill its budget.
      3. Run FPS on the remaining vertices (not yet selected) to fill global budget.

    Args:
        vertices: np.ndarray (N, 3) — mean face vertices
        kpt_ind:  np.ndarray (68,)  — dense indices of 68 sparse landmarks

    Returns:
        selected_indices: np.ndarray (468,) — indices into `vertices`
        region_map: dict mapping region name → list of positions within selected_indices
    """
    all_selected = []
    region_map = {}
    used_global = set()

    for region_name, sparse_seeds in SPARSE_REGION_SEEDS.items():
        budget = REGION_BUDGET[region_name]

        # Find dense vertices near this region's sparse seeds
        candidate_indices = _find_region_vertices(vertices, kpt_ind, sparse_seeds)

        # Exclude vertices already claimed by a previous region
        candidate_indices = np.array(
            [i for i in candidate_indices if i not in used_global]
        )

        if len(candidate_indices) == 0:
            print(f"  Warning: no candidate vertices found for {region_name}")
            region_map[region_name] = []
            continue

        # Run FPS within this region
        n_select = min(budget, len(candidate_indices))
        fps_local = farthest_point_sampling(vertices[candidate_indices], n_select)
        chosen = candidate_indices[fps_local]

        # Record: store position offsets relative to all_selected start
        start = len(all_selected)
        all_selected.extend(chosen.tolist())
        region_map[region_name] = list(range(start, start + len(chosen)))

        used_global.update(chosen.tolist())

    # Global fill: FPS on all vertices not yet selected
    remaining = np.array([i for i in range(len(vertices)) if i not in used_global])
    fps_global = farthest_point_sampling(vertices[remaining], GLOBAL_BUDGET)
    chosen_global = remaining[fps_global]
    region_map["global"] = list(range(len(all_selected), len(all_selected) + len(chosen_global)))
    all_selected.extend(chosen_global.tolist())

    print(f"  Total points selected: {len(all_selected)}")
    for r, idxs in region_map.items():
        print(f"    {r}: {len(idxs)} points")

    return np.array(all_selected, dtype=np.int64), region_map


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------

def build_and_save_point_index(seed=42):
    """
    Run FPS once, save the 468 indices and pointmap JSON files.
    Call this once from the repo root before running analyse_video.py.

    Args:
        seed: random seed for reproducibility (default 42)
    """
    np.random.seed(seed)
    print("Loading BFM mean face...")
    vertices, kpt_ind = load_bfm_mean_face()
    print(f"  Dense mesh: {len(vertices)} vertices")
    print(f"  Sparse landmarks: {len(kpt_ind)} keypoints")

    print("Running region-biased FPS...")
    indices, region_map = select_468_points(vertices, kpt_ind)

    # ── bfm_468_indices.json ──────────────────────────────────────────────
    # Simple lookup: just the 468 vertex indices.
    # Used by analyse_video.py every run.
    index_data = {
        "indices": indices.tolist(),    # list of 468 BFM vertex indices
        "region_map": region_map,       # which positions belong to which region
        "nose_index_in_dense": int(kpt_ind[NOSE_INDEX_68]),  # for nose subtraction
        "total_dense_vertices": len(vertices),
    }
    with open(INDEX_OUT, "w") as f:
        json.dump(index_data, f, indent=2)
    print(f"  Saved index file: {INDEX_OUT}")

    # ── bfm_468_pointmap.json ─────────────────────────────────────────────
    # Shareable reference: index + 3D coordinate on BFM mean face.
    # An InsightFace user loads this and finds nearest neighbors in their mesh.
    pointmap = []
    coords = vertices[indices]
    for i, (idx, xyz) in enumerate(zip(indices.tolist(), coords.tolist())):
        # Find which region this point belongs to
        region_label = "global"
        for region_name, positions in region_map.items():
            if i in positions:
                region_label = region_name
                break
        pointmap.append({
            "point_id": i,                   # 0-467, consistent across all users
            "bfm_vertex_index": idx,         # index into BFM dense mesh
            "mean_face_x": round(xyz[0], 4), # 3D position on mean face
            "mean_face_y": round(xyz[1], 4), # (use these to find correspondence
            "mean_face_z": round(xyz[2], 4), #  in InsightFace or other meshes)
            "region": region_label,
        })

    with open(POINTMAP_OUT, "w") as f:
        json.dump(pointmap, f, indent=2)
    print(f"  Saved pointmap file: {POINTMAP_OUT}")
    print("Done. Share bfm_468_pointmap.json with collaborators using other methods.")


def load_point_index(index_path=INDEX_OUT):
    """
    Load the saved 468-point index file.

    Returns:
        indices:      np.ndarray (468,) — BFM dense vertex indices
        region_map:   dict — region name → list of positions in indices array
        nose_dense:   int  — dense mesh index for nose tip (for nose subtraction)
    """
    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Point index file not found: {index_path}\n"
            "Run this first:\n"
            "  python -c \"from utils.fps_points import build_and_save_point_index; "
            "build_and_save_point_index()\""
        )
    with open(index_path) as f:
        data = json.load(f)

    return (
        np.array(data["indices"], dtype=np.int64),
        data["region_map"],
        int(data["nose_index_in_dense"]),
    )
