"""
utils/region_motion.py

Extracts per-region position, velocity, and acceleration from 3DDFA-V2's
68-point sparse landmarks.

Landmark index reference (standard 68-point scheme):
  Jaw outline  :  0-16
  Right eyebrow: 17-21
  Left eyebrow : 22-26
  Nose bridge  : 27-30  (30 = nose tip, used as reference)
  Nose base    : 31-35
  Right eye    : 36-41
  Left eye     : 42-47
  Outer lips   : 48-59
  Inner lips   : 60-67

Cheek note: the 68-point set has no true cheek landmarks. We use upper
jaw-contour points as the closest proxy:
  Right cheek  : 1, 2, 3   (right jaw, upper portion)
  Left cheek   : 13, 14, 15 (left jaw, upper portion)
These sit slightly low on the face but move with cheek dynamics well enough
for velocity/acceleration analysis.
"""

import numpy as np

# --- Region definitions ---------------------------------------------------

# Each region is a list of landmark indices.
# The centroid of these points is used as the region's position each frame.

REGIONS = {
    "right_eye":   list(range(36, 42)),   # 6 points
    "left_eye":    list(range(42, 48)),   # 6 points
    "lips":        list(range(48, 68)),   # 20 points (inner + outer)
    "right_cheek": [1, 2, 3],            # jaw-contour proxy
    "left_cheek":  [13, 14, 15],         # jaw-contour proxy
}

# Nose tip — used as the rigid head-motion reference (same role as in your
# MediaPipe pipeline). We average a small cluster around the tip for stability.
NOSE_REFERENCE_IDX = [28, 29, 30]  # nose bridge base + tip


# --- Per-frame extraction -------------------------------------------------

def get_region_positions(ver):
    """
    Extract the centroid (x, y, z) of each region from a single frame's
    68-point landmark array.

    Parameters
    ----------
    ver : np.ndarray, shape (3, 68)
        Landmark array from tddfa.recon_vers(..., dense_flag=False)[face_idx].

    Returns
    -------
    positions : dict[str -> np.ndarray shape (3,)]
        Centroid position of each region in 3D.
    nose_ref : np.ndarray shape (3,)
        Nose reference centroid for this frame.
    """
    positions = {}
    for name, indices in REGIONS.items():
        positions[name] = ver[:, indices].mean(axis=1)  # (3,)

    nose_ref = ver[:, NOSE_REFERENCE_IDX].mean(axis=1)  # (3,)
    return positions, nose_ref


def subtract_nose(positions, nose_ref):
    """
    Subtract nose reference from each region to get motion relative to the
    rigid head, cancelling out global head translation.

    Parameters
    ----------
    positions : dict[str -> np.ndarray (3,)]
    nose_ref  : np.ndarray (3,)

    Returns
    -------
    relative : dict[str -> np.ndarray (3,)]
    """
    return {name: pos - nose_ref for name, pos in positions.items()}


# --- Sequence processing --------------------------------------------------

def compute_kinematics(position_sequence, fps):
    """
    Compute velocity and acceleration from a sequence of positions using
    numpy.gradient (central differences internally, one-sided at edges).

    Parameters
    ----------
    position_sequence : np.ndarray, shape (N, 3)
        Centroid positions over N frames (after nose subtraction).
    fps : float
        Frame rate of the video in frames per second.

    Returns
    -------
    velocity     : np.ndarray, shape (N, 3)   units: position-units / second
    acceleration : np.ndarray, shape (N, 3)   units: position-units / second^2
    """
    dt = 1.0 / fps
    velocity     = np.gradient(position_sequence, dt, axis=0)
    acceleration = np.gradient(velocity,          dt, axis=0)
    return velocity, acceleration


def process_video_landmarks(ver_sequence, fps):
    """
    Full pipeline: given all per-frame landmark arrays for one face track,
    return positions, velocities, and accelerations for every region.

    Parameters
    ----------
    ver_sequence : list of np.ndarray (3, 68)
        One entry per video frame.
    fps : float
        Video frame rate.

    Returns
    -------
    results : dict with structure:
        results[region_name] = {
            "position"    : np.ndarray (N, 3),
            "velocity"    : np.ndarray (N, 3),
            "acceleration": np.ndarray (N, 3),
        }
        All positions are nose-subtracted (relative motion only).
    """
    # Collect raw positions and nose references across all frames
    raw_positions = {name: [] for name in REGIONS}
    nose_refs = []

    for ver in ver_sequence:
        positions, nose_ref = get_region_positions(ver)
        relative = subtract_nose(positions, nose_ref)
        nose_refs.append(nose_ref)
        for name in REGIONS:
            raw_positions[name].append(relative[name])

    # Convert to arrays and compute kinematics per region
    results = {}
    for name in REGIONS:
        pos_arr = np.array(raw_positions[name])          # (N, 3)
        vel, acc = compute_kinematics(pos_arr, fps)
        results[name] = {
            "position":     pos_arr,
            "velocity":     vel,
            "acceleration": acc,
        }

    return results
