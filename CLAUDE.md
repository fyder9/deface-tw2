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
- Planned: license plate detector
- Planned: person/head detectors (YOLO / RetinaNet class)

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

resize → stride-align → preprocess → ONNX inference → decode → scale-back → clamp → NMS → return dets → blur masks

Important constraints:

- SCRFD requires stride-aligned input (multiple of 32)
- scale factor must be tracked for coordinate restoration
- padded size is not equal to original size
- decode must use:
  - stride
  - feature map dimensions
  - anchor count
- NMS must operate on decoded pixel boxes
- mask scaling supported (mask-scale argument)

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

--detector <name>
--thresh <value>
--mask-scale <value>

Model variants supported:
- scrfd2.5g
- scrfd10g

Planned debug flag:

--scores → draw boxes + scores instead of blur

CLI args must propagate into detector behavior and must not be ignored.

---

## Known Problem Areas

Sensitive areas that must not be broken:

- ONNX shape mismatch when SCRFD input not stride-aligned
- decoding bugs from wrong feature map math
- anchor indexing errors
- score tensor dimensionality differences
- GPU provider misconfiguration
- CUDA DLL dependency issues
- DirectML vs CUDA confusion on Windows
- Conda environment mismatch issues
- model output layout differences between SCRFD variants
- padding vs scaling coordinate errors
- bounding box misalignment bugs

Claude must preserve correctness in:

- resize logic
- scale restoration
- decode math
- provider selection logic

---

## Future Model Expansion

Planned detector additions:

- license plate detector (ONNX)
- person detector (YOLO / RetinaNet ONNX)
- head detector
- multi-class anonymization modes

Detector abstraction must remain model-agnostic.

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