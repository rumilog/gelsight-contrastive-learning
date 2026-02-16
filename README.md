# GelSight Contrastive Learning - Billet Surface Data Collection

This repository contains tools for collecting tactile surface images from a **GelSight Mini** sensor, used for contrastive learning to distinguish between **unground (rough)** and **ground (smooth)** metal billet surfaces.

## Repository Overview

```
gsrobotics/
├── demo_billet_capture.py       # Main data collection script
├── demo_liveview.py             # Live view with device selection dropdown
├── demo_view3D.py               # 3D depth viewer (camera + contact mask + depth)
├── default_config.json          # Camera and model configuration
├── config.py                    # Config model class
├── models/
│   └── nnmini.pt                # Neural network for depth estimation
├── utilities/                   # Core library (camera, reconstruction, etc.)
├── billet_captures/             # Unground (rough) billet data
│   ├── billet1_camera_1.png     # Raw camera image
│   ├── billet1_depth_1.png      # Colorized depth image
│   ├── billet1_depth_1.npy      # Raw depth data (numpy float array)
│   └── ...
└── billet_captures_grinded/     # Ground (smooth) billet data
    ├── billet1_camera_1.png
    ├── billet1_depth_1.png
    ├── billet1_depth_1.npy
    └── ...
```

## Collected Data

### Unground samples (`billet_captures/`)
- 7 billets (billet1 through billet7)
- 342 files total (camera PNG + depth PNG + depth NPY per capture)

### Ground samples (`billet_captures_grinded/`)
- 5 billets (billet1 through billet5)
- 240 files total

### File naming convention

Each save produces 3 files:
- `billet{N}_camera_{M}.png` - Raw RGB camera image from the GelSight Mini
- `billet{N}_depth_{M}.png` - Colorized depth map (for visual inspection)
- `billet{N}_depth_{M}.npy` - Raw depth map as a numpy float array (for training)

Where `N` = billet number, `M` = image number (position on the billet).

## Getting Started on a New Machine

### 1. Clone the repo

```bash
git clone https://github.com/rumilog/gelsight-contrastive-learning.git
cd gelsight-contrastive-learning
```

### 2. Install dependencies (Python 3.10+)

```bash
python3.10 -m pip install --user -r requirements.txt
```

If `pip` is not installed for Python 3.10:
```bash
curl -sS https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3.10 get-pip.py --user
rm get-pip.py
python3.10 -m pip install --user -r requirements.txt
```

### 3. Loading the data for training

```python
import numpy as np
import os
from PIL import Image

# Load a single depth map (raw float data)
depth = np.load("billet_captures/billet1_depth_1.npy")
print(depth.shape, depth.dtype)  # (240, 320) float64

# Load a camera image
camera = np.array(Image.open("billet_captures/billet1_camera_1.png"))
print(camera.shape)  # (240, 320, 3)

# Load all unground depth maps
unground_depths = []
for f in sorted(os.listdir("billet_captures")):
    if f.endswith("_depth.npy") or (f.endswith(".npy")):
        unground_depths.append(np.load(os.path.join("billet_captures", f)))

# Load all ground depth maps
ground_depths = []
for f in sorted(os.listdir("billet_captures_grinded")):
    if f.endswith(".npy"):
        ground_depths.append(np.load(os.path.join("billet_captures_grinded", f)))

print(f"Unground samples: {len(unground_depths)}")
print(f"Ground samples: {len(ground_depths)}")
```

## Collecting More Data

If you need to collect additional samples, connect a GelSight Mini sensor and run:

### Unground samples
```bash
QT_QPA_PLATFORM_PLUGIN_PATH="" python3.10 demo_billet_capture.py
```

### Ground samples
```bash
QT_QPA_PLATFORM_PLUGIN_PATH="" python3.10 demo_billet_capture.py --output-dir ./billet_captures_grinded
```

### Capture controls

| Key | Action |
|-----|--------|
| **S** | Save camera + depth images (auto-increments image number) |
| **B** | Next billet (increments billet number, resets image count) |
| **D** | Cycle to next camera device |
| **R** | Reset image number to 1 |
| **Q** | Quit |

The script **auto-resumes** from wherever you left off - if you had billet 3 image 12, restarting will continue at billet 3 image 13.

### Finding the right camera device

If you're unsure which device index is the GelSight Mini, run:
```bash
python3.10 demo_liveview.py
```
This opens a GUI with a dropdown to cycle through all connected cameras. The GelSight Mini shows up as "Arducam" / "GelSight Mini" in the device list.

The default device index is set in `default_config.json` under `"default_camera_index"`.

## Configuration

Edit `default_config.json` to adjust settings:

```json
{
    "default_camera_index": 1,
    "camera_width": 320,
    "camera_height": 240,
    "border_fraction": 0.15,
    "marker_mask_min": 0,
    "marker_mask_max": 70,
    "nn_model_path": "./models/nnmini.pt",
    "use_gpu": false
}
```

Key settings:
- **default_camera_index**: Which camera to use (run `demo_liveview.py` to find the right one)
- **camera_width/height**: Resolution for depth estimation (320x240 default)
- **use_gpu**: Set to `true` if CUDA is available for faster depth inference

## Qt Plugin Error

If you see `Could not find the Qt platform plugin "xcb"`, prefix the command with:
```bash
QT_QPA_PLATFORM_PLUGIN_PATH="" python3.10 <script>
```

## Original GelSight SDK

This repo is based on the [GelSight Mini Python SDK](https://github.com/gelsightinc/gsrobotics). The `demo_billet_capture.py` script and data collection workflow were added for this project.
