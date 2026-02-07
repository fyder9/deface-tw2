# Project Context — Video Anonymization Tool (Deface Fork + ONNX GPU)

## Project Overview

This project is a fork and extension of the original “deface” tool for automatic anonymization of videos.

The goal is to build a professional-grade anonymization pipeline for law-enforcement usage, focused on:

- face detection + anonymization
- license plate detection + anonymization (in progress)
- future: head / person detection support
- GPU-accelerated inference with ONNX Runtime
- robust batch video processing (not real-time streaming)

Primary target users: law enforcement / police investigators processing recorded CCTV and bodycam footage.

This is not a demo tool. Design decisions favor correctness, determinism, and reproducibility over UI polish.

---

## Core Technical Stack

Backend:
- Python
- Fork of deface
- ONNX Runtime
- OpenCV
- imageio / ffmpeg pipeline
- Conda environments
- GPU execution via:
  - CUDAExecutionProvider (preferred)
  - DirectMLExecutionProvider (Windows fallback)
  - CPUExecutionProvider (last fallback)

Models:
- Original: CenterFace (face detection)
- Added: SCRFD (high-accuracy face detector)
- Added: YOLODetector (person detection with YOLOv4/v8)
- Planned: license plate detector
- Planned: head detectors (RetinaNet class)

Frontend:
- Node.js server wrapper
- React-based UI
- Upload → process → download workflow
- Planned case-oriented workflow (“create case” concept)

---

## Design Philosophy

Code philosophy:

- Do NOT hardcode model-specific constants unless required by model spec
- Keep detectors modular and swappable
- Match original deface detector interface:
  detector(frame, threshold) → dets, landmarks
- Preserve backward compatibility with CenterFace behavior
- Avoid changing existing deface logic unless strictly required
- Add features via flags and optional args — never break default flow
- Prefer ONNX models already converted
- Prefer GPU execution automatically when available
- Always allow CPU fallback without crash
- Keep preprocessing and decoding explicit and readable

---

## Detector Architecture Direction

Custom detector class introduced:

SCRFDdetector:
- ONNXRuntime session wrapper
- GPU provider selection logic
- dynamic provider fallback
- stride-aligned resize
- decode outputs → boxes + landmarks
- custom NMS
- returns CenterFace-compatible output format

Compatibility rule:

All detectors must return:

dets: Nx5 → [x1, y1, x2, y2, score]
lms: Nx10 → landmarks or zeros

So downstream blur pipeline stays unchanged.

---

## Current Detection Pipeline

Per frame pipeline:

resize → preprocess → ONNX inference → decode → scale-back → clamp → NMS → return dets → blur masks

Important constraints:

**SCRFD-specific:**
- requires stride-aligned input (multiple of 32)
- scale factor tracked for coordinate restoration
- padded size tracked separately

**YOLO-specific:**
- uses fixed 416×416 or 640×640 letterbox
- constant padding (128) instead of stride-align
- padding offsets (dw, dh) tracked for reversal
- filters by person class (COCO class 0) only

**Universal:**
- decode must use stride, feature map dimensions, anchor count
- NMS must operate on decoded pixel boxes
- mask scaling supported (--mask-scale argument)
- all detectors return (Nx5 dets, Nx10 lms) format

---

## Model Storage & Organization

All ONNX models are stored in centralized `/models` directory:

```
project-root/
├── models/
│   ├── scrfd_2.5g.onnx (3.1 MB)
│   ├── scrfd_10g.onnx (15.5 MB)
│   ├── yolov4.onnx (246 MB)
│   ├── yolov4-tiny.onnx (23 MB)
│   ├── yolov8.onnx (167 MB)
│   └── centerface.onnx (7.0 MB)
├── deface/
└── [other files]
```

Model path resolution:
- Files in deface/ use: `os.path.dirname(os.path.dirname(__file__)) / models / model.onnx`
- Relative path traversal: deface/ → project_root/ → models/
- Cross-platform compatible via `os.path.join()`

---

## GPU Execution Requirements

ONNX Runtime GPU requires:

CUDAExecutionProvider:
- CUDA Toolkit 12.x
- cuBLAS
- cuDNN
- correct PATH entries
- matching onnxruntime-gpu build

Fallback order implemented:

CUDA → DirectML → CPU

The code must print:

- available providers
- selected providers
- active provider

These debug prints must not be removed.

---

## CLI Extensions Added

CLI extensions added:

--detector <name>           # Choose detector backend
--thresh <value>            # Detection confidence threshold
--mask-scale <value>        # Scale factor for masks
--yolo-variant <name>       # YOLO model variant (v4 or v8)

Detector choices:
- scrfd2.5g (face detection, 3.1 MB)
- scrfd10g (face detection, 15.5 MB)
- centerface (face detection, 7.0 MB)
- yolo (person detection, YOLOv4 or YOLOv8)

YOLO variants:
- v4 (default, 246 MB, faster)
- v8 (experimental, 167 MB, newer architecture)

Debug flags:

--scores → draw boxes + scores instead of blur

CLI args must propagate into detector behavior and must not be ignored.

---

## Known Problem Areas

Sensitive areas that must not be broken:

**SCRFD-specific:**
- ONNX shape mismatch when input not stride-aligned
- padding vs scaling coordinate errors
- model output layout differences between SCRFD 2.5G and 10G

**YOLO-specific:**
- letterbox padding offset calculation (dw, dh must be integer divided by 2)
- sigmoid + xy_scale formula in bbox decoding
- exp(wh) * anchor multiplication order
- class filtering (person class only, COCO class 0)
- coordinate reversal after letterbox removal
- YOLOv4 vs YOLOv8 output shape differences

**Universal:**
- decoding bugs from wrong feature map math
- anchor indexing errors
- score tensor dimensionality differences
- GPU provider misconfiguration
- CUDA DLL dependency issues
- DirectML vs CUDA confusion on Windows
- Conda environment mismatch issues
- bounding box misalignment bugs

Claude must preserve correctness in:

- resize logic (stride-aligned vs letterbox)
- scale restoration
- decode math (SCRFD offset vs YOLO sigmoid/exp)
- provider selection logic
- padding offset reversal

---

## Detector Requirements & Specifications

All detectors must:
1. Accept frame in BGR (OpenCV format)
2. Return `(dets, lms)` tuple:
   - dets: (N, 5) array [x1, y1, x2, y2, score]
   - lms: (N, 10) array [landmarks or zeros]
3. Support threshold parameter in `__call__(frame, threshold)`
4. Print GPU provider selection at init
5. Handle GPU fallback (CUDA → DirectML → CPU)
6. Preserve coordinate space (original frame dimensions)

**YOLO Detector specifics:**
- Input: RGB, normalized [0, 1], letterboxed to fixed size
- Output: 3 scales with anchor predictions
- Filters person class only (COCO class 0)
- No landmarks (returns zeros for compatibility)
- Supports multiple variants via --yolo-variant arg

## Future Model Expansion

Planned detector additions:

- license plate detector (ONNX)
- head detector (RetinaNet ONNX)
- multi-class anonymization modes

Detector abstraction must remain model-agnostic and follow specs above.

---

## Code Quality Expectations

When modifying code:

- do not rewrite architecture
- do not break CLI compatibility
- do not remove fallback paths
- do not introduce hidden constants
- prefer explicit math over implicit assumptions
- keep decode math readable
- preserve debug prints
- avoid silent behavior changes
- avoid hardcoded shapes unless model requires it

---

## Operational Goal

Final tool should be:

- deterministic
- GPU accelerated
- batch capable
- legally deployable
- modular
- explainable
- robust on CCTV footage
- easy to audit

Priority order:

correct anonymization > stability > performance > elegance