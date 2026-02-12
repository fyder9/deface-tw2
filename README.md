[![PyPI](https://img.shields.io/pypi/v/deface)](https://pypi.org/project/deface/) [![GitHub Workflow Status](https://img.shields.io/github/actions/workflow/status/ORB-HD/deface/python-publish.yml)](https://github.com/ORB-HD/deface/actions)

# `deface`: Video anonymization by face detection

`deface` is a simple command-line tool for automatic anonymization of faces in videos or photos.
It works by first detecting all human faces in each video frame and then applying an anonymization filter (blurring or black boxes) on each detected face region.
By default all audio tracks are discarded as well.


Original frame | `deface` output (using default options)
:--:|:--:
![examples/city.jpg](examples/city.jpg) | ![$ deface examples/city.jpg](examples/city_anonymized.jpg)


## Installation

`deface` supports all commonly used operating systems (Linux, Windows, MacOS), but it requires using a command-line shell such as bash. There are currently no plans of creating a graphical user interface.

The recommended way of installing `deface` is via the `pip` package manager. This requires that you have Python 3.6 or later installed on your system. It is recommended to set up and activate a new [virtual environment](https://realpython.com/python-virtual-environments-a-primer/) first. Then you can install the latest release of `deface` and all necessary dependencies by running:

    $ python3 -m pip install deface

Alternatively, if you want to use the latest (unreleased) revision directly from GitHub, you can run:

    $ python3 -m pip install 'git+https://github.com/ORB-HD/deface'

This will only install the dependencies that are strictly required for running the tool. If you want to speed up processing by enabling hardware acceleration, you will need to manually install additional packages, see [Hardware acceleration](#hardware-acceleration)


## Usage

### Quick start

If you want to try out anonymizing a video using the default settings, you just need to supply the path to it. For example, if the path to your test video is `myvideos/vid1.mp4`, run:

    $ deface myvideos/vid1.mp4

This will write the the output to the new video file `myvideos/vid1_anonymized.mp4`.

### Live capture demo

If you have a camera (webcam) attached to your computer, you can run `deface` on the live video input by calling it with the `cam` argument instead of an input path:

    $ deface cam

This is a shortcut for `$ deface --preview '<video0>'`, where `'<video0>'` (literal) is a  camera device identifier. If you have multiple cameras installed, you can try `'<videoN>'`, where `N` is the index of the camera (see [imageio-ffmpeg docs](https://imageio.readthedocs.io/en/stable/format_ffmpeg.html)).

### CLI usage and options summary

To get an overview of usage and available options, run:

    $ deface -h

The full list of command-line arguments is comprehensive and includes detection, tracking, and anonymization options. See sections below for detailed information on specific features.

## Usage examples

In most use cases the default configuration should be sufficient, but depending on individual requirements and type of media to be processed, some of the options might need to be adjusted. In this section, some common example scenarios that require option changes are presented. All of the examples use the photo [examples/city.jpg](examples/city.jpg), but they work the same on any video or photo file.

### Drawing black boxes

By default, each detected face is anonymized by applying a blur filter to an ellipse region that covers the face. If you prefer to anonymize faces by drawing black boxes on top of them, you can achieve this through the `--boxes` and `--replacewith` options:

    $ deface examples/city.jpg --boxes --replacewith solid -o examples/city_anonymized_boxes.jpg

<img src="examples/city_anonymized_boxes.jpg" width="70%" alt="$ deface examples/city.jpg --enable-boxes --replacewith solid -o examples/city_anonymized_boxes.jpg"/>

### Mosaic anonymization

Another common anonymization option is to draw a mosaic pattern over faces. This is supported with the `--replacewith mosaic` option. The width of each of the quadratic mosaic fragments can be determined using the `--mosaicsize` option (default value: 20). Note that the mosaic size is measured in pixels, so you should consider increasing the size when processing higher-resolution inputs.

Usage example:

    $ deface examples/city.jpg --replacewith mosaic --mosaicsize 20 -o examples/city_anonymized_mosaic.jpg

<img src="examples/city_anonymized_mosaic.jpg" width="70%" alt="$ deface examples/city.jpg --replacewith mosaic --mosaicsize 20 -o examples/city_anonymized_mosaic.jpg"/>



### Tuning detection thresholds

The detection threshold (`--thresh`, `-t`) is used to define how confident the detector needs to be for classifying some region as a face. By default this is set to the value 0.2, which was found to work well on many test videos.

If you are experiencing too many false positives (i.e. anonymization filters applied at non-face regions) on your own video data, consider increasing the threshold.
On the other hand, if there are too many false negative errors (visible faces that are not anonymized), lowering the threshold is advisable.

The optimal value can depend on many factors such as video quality, lighting conditions and prevalence of partial occlusions. To optimize this value, you can set threshold to a very low value and then draw detection score overlays, as described in the [section below](#drawing-detection-score-overlays).

To demonstrate the effects of a threshold that is set too low or too high, see the examples outputs below:

`--thresh 0.02` (notice the false positives, e.g. at hand regions) | `--thresh 0.7` (notice the false negatives, especially at partially occluded faces)
:--:|:--:
![examples/city_anonymized_thresh0.02.jpg](examples/city_anonymized_thresh0.02.jpg) | ![$ deface examples/city_anonymized_thresh0.7.jpg](examples/city_anonymized_thresh0.7.jpg)


### Drawing detection score overlays

If you are interested in seeing the faceness score (a score between 0 and 1 that roughly corresponds to the detector's confidence that something *is* a face) of each detected face in the input, you can enable the `--draw-scores` option to draw the score of each detection directly above its location.

    $ deface examples/city.jpg --draw-scores -o examples/city_anonymized_scores.jpg

<img src="examples/city_anonymized_scores.jpg" width="70%" alt="$ deface examples/city.jpg --draw-scores -o examples/city_anonymized_scores.jpg"/>

This option can be useful to figure out an optimal value for the detection threshold that can then be set through the `--thresh` option.


### High-resolution media and performance issues

Since `deface` tries to detect faces in the unscaled full-res version of input files by default, this can lead to performance issues on high-res inputs (>> 720p). In extreme cases, even detection accuracy can suffer because the detector neural network has not been trained on ultra-high-res images.

To counter these performance issues, `deface` supports downsampling its inputs on-the-fly before detecting faces, and subsequently rescaling detection results to the original resolution. Downsampling only applies to the detection process, whereas the final output resolution remains the same as the input resolution.

This feature is controlled through the `--scale` option, which expects a value of the form `WxH`, where `W` and `H` are the desired width and height of downscaled input representations.
It is very important to make sure the aspect ratio of the inputs remains intact when using this option, because otherwise, distorted images are fed into the detector, resulting in decreased accuracy.

For example, if your inputs have the common aspect ratio 16:9, you can instruct the detector to run in 360p resolution by specifying `--scale 640x360`.
If the results at this fairly low resolution are not good enough, detection at 720p input resolution (`--scale 1280x720`) may work better.


## Hardware acceleration

Depending on your available hardware, you can speed up neural network inference by enabling the optional [ONNX Runtime](https://microsoft.github.io/onnxruntime/) backend of `deface`. For optimal performance you should install it with appropriate [Execution Providers](https://onnxruntime.ai/docs/execution-providers) for your system. If you have multiple Execution Providers installed, ONNX Runtime will try to automatically use the fastest one available.

Here are some recommendations for common setups:

### CUDA (only for Nvidia GPUs)

If you have a CUDA-capable GPU, you can enable GPU acceleration by installing the relevant packages:

    $ python3 -m pip install onnx onnxruntime-gpu

If the `onnxruntime-gpu` package is found and a GPU is available, the face detection network is automatically offloaded to the GPU.
This can significantly improve the overall processing speed.

### DirectML (only for Windows)

Windows users with capable non-Nvidia GPUs can enable GPU-accelerated inference with DirectML by installing:

    $ python3 -m pip install onnx onnxruntime-directml

### OpenVINO

OpenVINO can accelerate inference even on CPU-only systems by a few percent, compared to the default OpenCV and ONNX Runtime implementations. It works on Linux and Windows, but not yet on Python 3.11 as of July 2023. Install the backend with:

    $ python3 -m pip install onnx onnxruntime-openvino


### Other platforms

If you your setup doesn't fit with these recommendations, look into the available options at the [Execution Provider](https://onnxruntime.ai/docs/execution-providers/#summary-of-supported-execution-providers) documentation and find the respective installation instructions in the [ONNX Runtime build matrix](https://microsoft.github.io/onnxruntime/).


## Face & Person Detection

`deface` supports multiple detection backends. Select with `--detector`:

```bash
deface video.mp4 --detector <detector_name>
```

### Available Detectors

| Detector | Size | Type | Speed | Best For |
|----------|------|------|-------|----------|
| `scrfd2.5g` (default) | 3.1 MB | Face | Very fast | Quick processing, low resources |
| `scrfd10g` | 15.5 MB | Face | Fast | Best accuracy (production) |
| `centerface` | 7.0 MB | Face | Fast | Legacy compatibility |
| `yolo` | 246 MB | Person/body | Moderate | Full body detection |
| `yolo --yolo-variant v8` | 167 MB | Person/body | Fast | Modern person detection |
| `crowdhuman` | 80 MB | Person/head | Fast | Dense crowds, occlusions |

**Threshold tuning** (`--thresh`, default 0.3):
- Lower = more detections (more false positives)
- Higher = fewer detections (more false negatives)
- Use `--scores --preview` to visualize and tune

**Example**: Detect crowds with visualization
```bash
deface video.mp4 --detector crowdhuman --thresh 0.4 --scores --preview
```

---

## Temporal Tracking

Enable temporal face tracking to maintain consistent anonymization across frames (useful for law enforcement CCTV):

```bash
deface video.mp4 --enable-tracking
```

**How it works**: Tracks faces across frames and fills gaps during temporary occlusions. Prevents flickering and maintains identity consistency.

**Arguments**:
```bash
--enable-tracking               # Enable tracking
--track-iou-threshold 0.3       # IoU threshold for matching (higher = stricter)
--track-dist-threshold 50.0     # Max distance for matching (pixels)
--track-alpha 0.3               # Smoothing (0 = off, 1 = full smoothing)
--track-ttl 10                  # Frames to keep blurring after detection loss
--track-expansion 0.05          # Box expansion per missed frame (5% per side)
--track-debug                   # Print debug info
```

**Examples**:
```bash
# Strict matching (fewer tracking errors):
deface video.mp4 --enable-tracking --track-iou-threshold 0.4 --track-ttl 5

# Loose matching (longer memory for occlusions):
deface video.mp4 --enable-tracking --track-iou-threshold 0.2 --track-ttl 20

# Smooth motion:
deface video.mp4 --enable-tracking --track-alpha 0.6
```

---

## Proximity Search (ROI Re-detection)

Re-scan regions near recently-seen faces to catch detections missed by the full-frame detector:

```bash
deface video.mp4 --enable-proximity-search
```

**How it works**: Crops regions around last-known faces and runs detector on those ROIs. Catches small, occluded, or edge-of-frame faces.

**Arguments**:
```bash
--enable-proximity-search               # Enable proximity search
--proximity-ttl 15                      # Max age of face box for ROI seeding (frames)
--proximity-expand 2.0                  # ROI expansion (2.0 = 2× box size)
--proximity-thresh <value>              # Min confidence for proximity detections
--proximity-iou 0.1                     # Min IoU for position validation
--proximity-dist 0.5                    # Max center distance (normalized)
--proximity-area-min 10                 # Min box area to seed ROI (pixels)
--proximity-area-max 10000              # Max box area to seed ROI
--proximity-debug                       # Print debug info
```

**Examples**:
```bash
# Conservative (strict validation, small ROI):
deface video.mp4 --enable-proximity-search --proximity-expand 1.5 --proximity-iou 0.2

# Aggressive (loose validation, large ROI):
deface video.mp4 --enable-proximity-search --proximity-expand 3.0 --proximity-iou 0.05

# Combined with tracking:
deface video.mp4 --enable-tracking --enable-proximity-search --track-ttl 15

# Debug mode:
deface video.mp4 --enable-proximity-search --proximity-debug --scores --preview
```

---

## Command-Line Arguments Summary

**Input/Output**:
```
input                   File path(s), directory, or 'cam' for webcam
-o, --output FILE       Output file (default: input + "_anonymized")
-k, --keep-audio        Keep audio track from input (videos only)
```

**Detection**:
```
--detector {scrfd2.5g,scrfd10g,centerface,yolo,crowdhuman}
                        Detector choice (default: scrfd2.5g)
-t, --thresh T          Detection threshold, 0.0-1.0 (default: 0.3)
--yolo-variant {v4,v8}  YOLO variant (only with --detector yolo)
--scores                Show detection boxes/scores instead of blurring
```

**Anonymization**:
```
--replacewith {blur,solid,none,img,mosaic}
                        Filter type (default: blur)
--boxes                 Use rectangular boxes instead of ellipse masks
--mask-scale M          Mask scale factor (default: 1.3)
--mosaicsize WIDTH      Mosaic size in pixels (default: 20, for --replacewith mosaic)
--replaceimg IMAGE      Custom image for replacement (for --replacewith img)
```

**Display**:
```
-p, --preview           Live preview during processing
--draw-scores           Draw scores on anonymized output
--disable-progress-output   Hide progress bar
```

**Performance**:
```
-s, --scale WxH         Downscale for detection (e.g., 640x360)
--backend {auto,onnxrt,opencv}
                        Execution backend (default: auto)
-ep, --execution-provider EP
                        Force specific GPU provider (e.g., CUDAExecutionProvider)
--ffmpeg-config JSON    FFmpeg encoding options
```

**Tracking** (see [Temporal Tracking](#temporal-tracking) section):
```
--enable-tracking
--track-iou-threshold VALUE       (default: 0.3)
--track-dist-threshold VALUE      (default: 50.0)
--track-alpha VALUE               (default: 0.3)
--track-ttl FRAMES                (default: 10)
--track-expansion VALUE           (default: 0.05)
--track-debug
```

**Proximity Search** (see [Proximity Search](#proximity-search-roi-re-detection) section):
```
--enable-proximity-search
--proximity-ttl FRAMES            (default: 15)
--proximity-expand FACTOR         (default: 2.0)
--proximity-thresh VALUE
--proximity-iou VALUE             (default: 0.1)
--proximity-dist VALUE            (default: 0.5)
--proximity-area-min PIXELS       (default: 10)
--proximity-area-max PIXELS       (default: 10000)
--proximity-debug
```

**Other**:
```
-h, --help              Show full help message
--version               Show version
```

---

## How It Works

### Detection

The default detector is **SCRFD** ([paper](https://arxiv.org/abs/2105.04714)), an anchor-free face detector optimized for speed and accuracy. SCRFD is available in two sizes (2.5G and 10G) with different accuracy/speed tradeoffs.

Alternative detectors:
- **CenterFace**: Original upstream detector ([paper](https://arxiv.org/abs/1911.03599))
- **YOLO variants**: Person/body detection (YOLOv4, YOLOv8, CrowdHuman YOLOv5m)

### Processing Pipeline

For each frame:
1. Preprocess (resize, normalize)
2. Run neural network inference
3. Post-process (NMS, threshold, coordinate conversion)
4. Return bounding boxes
5. Apply anonymization filter

### Optional Features

- **Tracking**: Maintains consistent anonymization across frames (gap-filling during occlusions)
- **Proximity Search**: Re-detects faces in ROIs near recent detections
- **Visualization**: Display detection scores for parameter tuning (`--scores`)


## Credits

- `centerface.py` is based on https://github.com/Star-Clouds/centerface (revision [8c39a49](https://github.com/Star-Clouds/CenterFace/tree/8c39a497afb78fb2c064eb84bf010c273bb7d3ce)),
  [released under MIT license](https://github.com/Star-Clouds/CenterFace/blob/36afed/LICENSE)
- The included model file `centerface.onnx` is an unmodified copy of the [`centerface_bnmerged.onnx`](https://github.com/Star-Clouds/CenterFace/blob/b82ec0c4844e89fd5a0305986aed9bdf33c72585/models/onnx/centerface_bnmerged.onnx) from https://github.com/Star-Clouds/centerface
- The original source of the example images in the `examples` directory can be found [here](https://www.pexels.com/de-de/foto/stadt-kreuzung-strasse-menschen-109919/) (released under the [Pexels photo license](https://www.pexels.com/photo-license/))
