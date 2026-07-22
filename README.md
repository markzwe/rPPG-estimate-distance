# Facial Motion & Distance Estimation

Monocular video pipeline that reconstructs a dense 3D face, selects a canonical **468-point** landmark set with region-aware Farthest Point Sampling (FPS), and exports facial motion, head pose, and optional **metric depth**.

Built for research on facial dynamics and remote sensing distance cues (rPPG-related distance context).

---

## Highlights

- Dense 3D face reconstruction with **3DDFA-V2**
- Region-aware FPS → stable **468** landmarks (from ~38k BFM vertices)
- Nose-normalized regional motion (eyes, lips, cheeks)
- Head pose (yaw / pitch / roll)
- Optional metric depth via **UniDepth-V2** at each landmark
- CSV exports ready for analysis

---

## Pipeline

```
Image sequence
      │
      ▼
 Face detection (FaceBoxes or SCRFD)
      │
      ▼
 3DDFA dense reconstruction (~38,365 vertices)
      │
      ▼
 Region-aware FPS → 468 canonical landmarks
      │
      ├──────────────┐
      ▼              ▼
 Motion / pose    UniDepth (optional)
      │              │
      └──────┬───────┘
             ▼
        CSV outputs
```

### Landmark budget

| Region | Points |
|--------|--------|
| Right eye | 40 |
| Left eye | 40 |
| Lips | 60 |
| Right cheek | 30 |
| Left cheek | 30 |
| Global fill | 268 |
| **Total** | **468** |

---

## Repository layout

```
.
├── analyse_video.py          # main pipeline
├── demo_unidepth_faces.py    # InsightFace + UniDepth depth demo
├── TDDFA.py / TDDFA_ONNX.py
├── FaceBoxes/                # face detector
├── bfm/                      # BFM helpers
├── models/                   # 3DDFA backbones
├── weights/                  # pretrained 3DDFA checkpoints
├── configs/                  # 3DDFA + UniDepth configs
├── unidepth/                 # UniDepth model code
├── utils/                    # FPS, pose, region motion, …
├── bfm_468_indices.json      # precomputed FPS index
└── bfm_468_pointmap.json     # shareable 3D correspondence map
```

---

## Setup

```bash
git clone https://github.com/<your-username>/rPPG-Distance-Estimation.git
cd rPPG-Distance-Estimation

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -r requirements.txt
pip install -e .            # installs the local unidepth package
```

Notes:
- Python **3.10+** recommended
- First `--onnx` run auto-converts ONNX models from the shipped `.pth` / `.pkl` files
- Optional macOS GPU: PyTorch MPS is used automatically when available
- `triton` / `xformers` are **not** required for inference on macOS/CPU

If you need mesh rendering extras later:

```bash
cd Sim3DR && ./build_sim3dr.sh && cd ..
```

---

## Quick start

Input should be a folder of frames (`.png`, `.jpg`, …).

```bash
# Motion + pose
python analyse_video.py -f path/to/frames --onnx

# Stronger detector for distant faces
python analyse_video.py -f path/to/frames --onnx --detector scrfd

# Add UniDepth metric depth at the 468 landmarks
python analyse_video.py -f path/to/frames --onnx --with_depth
```

Depth-only InsightFace demo (optional):

```bash
# edit FOLDER_PATH inside the script, then:
python demo_unidepth_faces.py
```

### Regenerate the 468-point index (optional)

Already included in the repo. Rebuild with:

```bash
python -c "from utils.fps_points import build_and_save_point_index; build_and_save_point_index()"
```

---

## Outputs

Written next to the input folder name (prefix = folder basename), e.g. `frames_landmarks.csv`:

| File | Description |
|------|-------------|
| `*_landmarks.csv` | Nose-normalized 468 landmark XYZ |
| `*_landmarks_depth.csv` | Landmarks + projected pixels + depth |
| `*_pose.csv` | Yaw, pitch, roll |
| `*_right_eye.csv` / `*_left_eye.csv` | Regional kinematics |
| `*_lips.csv` | Lip motion |
| `*_right_cheek.csv` / `*_left_cheek.csv` | Cheek motion |

---

## Method (short)

1. Detect face (FaceBoxes or SCRFD)
2. Fit 3DMM with 3DDFA → dense BFM mesh
3. Index the precomputed region-aware FPS subset (468 points)
4. Nose-normalize coordinates; compute velocity / acceleration per region
5. Estimate head pose from 3DMM parameters
6. Optionally sample UniDepth at projected landmark pixels

---

## Tech stack

Python · OpenCV · NumPy · PyTorch · 3DDFA-V2 · BFM · UniDepth-V2 · InsightFace (optional) · ONNX Runtime

---

## Author

**Zwe Htet**  
University of Dayton · Department of Computer Science

---

## Acknowledgements

This project builds on:

- [3DDFA_V2](https://github.com/cleardusk/3DDFA_V2) (MIT)
- [Basel Face Model](https://faces.dmi.unibas.ch/bfm/) (academic use)
- [UniDepth](https://github.com/lpiccinelli-eth/UniDepth) (CC BY-NC 4.0)
- [InsightFace](https://github.com/deepinsight/insightface)

See [`NOTICE.md`](NOTICE.md) for license details.

---

## Citation

If this repository is useful in your work, please also cite the upstream papers for 3DDFA-V2, BFM, and UniDepth.
# rPPG-estimate-distance
