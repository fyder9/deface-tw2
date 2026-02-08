#!/usr/bin/env python3

import argparse
import json
import mimetypes
import os
from typing import Dict, Tuple, TYPE_CHECKING, Any, Optional

import tqdm
import skimage.draw
import numpy as np
import imageio
import imageio.v2 as iio
import cv2

from deface import __version__

if TYPE_CHECKING:
    from deface.centerface import CenterFace

# Models directory relative to project root
_project_root = os.path.dirname(os.path.dirname(__file__))
_models_dir = os.path.join(_project_root, 'models')
default_scrfd_onnx_path = os.path.join(_models_dir, 'scrfd_2.5g.onnx')


def scale_bb(x1, y1, x2, y2, mask_scale=1.0):
    s = mask_scale - 1.0
    h, w = y2 - y1, x2 - x1
    y1 -= h * s
    y2 += h * s
    x1 -= w * s
    x2 += w * s
    return np.round([x1, y1, x2, y2]).astype(int)


def draw_det(
        frame, score, det_idx, x1, y1, x2, y2,
        replacewith: str = 'blur',
        ellipse: bool = True,
        draw_scores: bool = False,
        ovcolor: Tuple[int] = (0, 0, 0),
        replaceimg = None,
        mosaicsize: int = 20,
        is_tracked: bool = False
):
    if replacewith == 'solid':
        cv2.rectangle(frame, (x1, y1), (x2, y2), ovcolor, -1)
    elif replacewith == 'blur':
        bf = 2  # blur factor (number of pixels in each dimension that the face will be reduced to)
        blurred_box =  cv2.blur(
            frame[y1:y2, x1:x2],
            (abs(x2 - x1) // bf, abs(y2 - y1) // bf)
        )
        if ellipse:
            roibox = frame[y1:y2, x1:x2]
            # Get y and x coordinate lists of the "bounding ellipse"
            ey, ex = skimage.draw.ellipse((y2 - y1) // 2, (x2 - x1) // 2, (y2 - y1) // 2, (x2 - x1) // 2)
            roibox[ey, ex] = blurred_box[ey, ex]
            frame[y1:y2, x1:x2] = roibox
        else:
            frame[y1:y2, x1:x2] = blurred_box
    elif replacewith == 'img':
        target_size = (x2 - x1, y2 - y1)
        resized_replaceimg = cv2.resize(replaceimg, target_size)
        if replaceimg.shape[2] == 3:  # RGB
            frame[y1:y2, x1:x2] = resized_replaceimg
        elif replaceimg.shape[2] == 4:  # RGBA
            frame[y1:y2, x1:x2] = frame[y1:y2, x1:x2] * (1 - resized_replaceimg[:, :, 3:] / 255) + resized_replaceimg[:, :, :3] * (resized_replaceimg[:, :, 3:] / 255)
    elif replacewith == 'mosaic':
        for y in range(y1, y2, mosaicsize):
            for x in range(x1, x2, mosaicsize):
                pt1 = (x, y)
                pt2 = (min(x2, x + mosaicsize - 1), min(y2, y + mosaicsize - 1))
                color = (int(frame[y, x][0]), int(frame[y, x][1]), int(frame[y, x][2]))
                cv2.rectangle(frame, pt1, pt2, color, -1)
    elif replacewith == 'none':
        # When in scores debug mode, draw bounding box rectangle
        if draw_scores:
            color = (0, 255, 255) if is_tracked else (0, 255, 0)  # Yellow for tracked, green for fresh
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    if draw_scores:
        color = (0, 255, 255) if is_tracked else (0, 255, 0)  # Yellow for tracked, green for fresh
        label = f'{score:.2f} (track)' if is_tracked else f'{score:.2f}'
        cv2.putText(
            frame, label, (x1 + 0, y1 - 20),
            cv2.FONT_HERSHEY_DUPLEX, 0.5, color
        )


def anonymize_frame(
        dets, frame, mask_scale,
        replacewith, ellipse, draw_scores, replaceimg, mosaicsize,
        original_dets_count: int = None
):
    for i, det in enumerate(dets):
        boxes, score = det[:4], det[4]
        x1, y1, x2, y2 = boxes.astype(int)
        x1, y1, x2, y2 = scale_bb(x1, y1, x2, y2, mask_scale)
        # Clip bb coordinates to valid frame region
        y1, y2 = max(0, y1), min(frame.shape[0] - 1, y2)
        x1, x2 = max(0, x1), min(frame.shape[1] - 1, x2)

        # Determine if this detection is gap-filled (tracked)
        is_tracked = (original_dets_count is not None) and (i >= original_dets_count)

        draw_det(
            frame, score, i, x1, y1, x2, y2,
            replacewith=replacewith,
            ellipse=ellipse,
            draw_scores=draw_scores,
            replaceimg=replaceimg,
            mosaicsize=mosaicsize,
            is_tracked=is_tracked
        )


def cam_read_iter(reader):
    while True:
        yield reader.get_next_data()


def video_detect(
        ipath: str,
        opath: str,
        centerface: Any,
        threshold: float,
        enable_preview: bool,
        cam: bool,
        nested: bool,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        ffmpeg_config: Dict[str, str],
        replaceimg = None,
        keep_audio: bool = False,
        mosaicsize: int = 20,
        disable_progress_output = False,
        enable_tracking: bool = False,
        track_iou_threshold: float = 0.25,
        track_dist_threshold: float = 1.5,
        track_alpha: float = 0.65,
        track_ttl: int = 10,
        track_expansion: float = 0.05,
        track_debug: bool = False,
        enable_proximity_search: bool = False,
        proximity_ttl: int = 10,
        proximity_expand: float = 1.8,
        proximity_thresh: Optional[float] = None,
        proximity_iou: float = 0.15,
        proximity_dist: float = 1.2,
        proximity_area_min: float = 0.4,
        proximity_area_max: float = 2.5,
        proximity_debug: bool = False
):
    try:
        if 'fps' in ffmpeg_config:
            reader: imageio.plugins.ffmpeg.FfmpegFormat.Reader = imageio.get_reader(ipath, fps=ffmpeg_config['fps'])
        else:
            reader: imageio.plugins.ffmpeg.FfmpegFormat.Reader = imageio.get_reader(ipath)

        meta = reader.get_meta_data()
        _ = meta['size']
    except:
        if cam:
            print(f'Could not find video device {ipath}. Please set a valid input.')
        else:
            print(f'Could not open file {ipath} as a video file with imageio. Skipping file...')
        return

    if cam:
        nframes = None
        read_iter = cam_read_iter(reader)
    else:
        read_iter = reader.iter_data()
        nframes = reader.count_frames()
    if nested:
        bar = tqdm.tqdm(dynamic_ncols=True, total=nframes, position=1, leave=True, disable=disable_progress_output)
    else:
        bar = tqdm.tqdm(dynamic_ncols=True, total=nframes, disable=disable_progress_output)

    if opath is not None:
        _ffmpeg_config = ffmpeg_config.copy()
        #  If fps is not explicitly set in ffmpeg_config, use source video fps value
        _ffmpeg_config.setdefault('fps', meta['fps'])
        # Carry over audio from input path, use "copy" codec (no transcoding) by default
        if keep_audio and meta.get('audio_codec'):
            _ffmpeg_config.setdefault('audio_path', ipath)
            _ffmpeg_config.setdefault('audio_codec', 'copy')
        writer: imageio.plugins.ffmpeg.FfmpegFormat.Writer = imageio.get_writer(
            opath, format='FFMPEG', mode='I', **_ffmpeg_config
        )

    # Initialize tracking if enabled
    frame_idx = 0
    tracker = None
    if enable_tracking:
        from deface.tracker import FaceTracker
        tracker = FaceTracker(
            iou_threshold=track_iou_threshold,
            norm_dist_threshold=track_dist_threshold,
            alpha=track_alpha,
            ttl=track_ttl,
            expansion_rate=track_expansion,
            debug=track_debug
        )
        if not disable_progress_output:
            print(f"[Tracking] Enabled with TTL={track_ttl}, IoU={track_iou_threshold}, "
                  f"NormDist={track_dist_threshold}, Alpha={track_alpha}, Expansion={int(track_expansion*100)}%")

    # Initialize proximity search if enabled
    proximity_search_mgr = None
    if enable_proximity_search:
        from deface.proximity_search import ProximitySearchManager, ProximitySearchConfig
        # Use 0.8 * threshold if not specified
        _proximity_thresh = proximity_thresh if proximity_thresh is not None else threshold * 0.8
        config = ProximitySearchConfig(
            proximity_ttl=proximity_ttl,
            proximity_expand=proximity_expand,
            proximity_thresh=_proximity_thresh,
            proximity_iou=proximity_iou,
            proximity_dist=proximity_dist,
            proximity_area_min=proximity_area_min,
            proximity_area_max=proximity_area_max,
            debug=proximity_debug
        )
        proximity_search_mgr = ProximitySearchManager(centerface, config)
        if not disable_progress_output:
            print(f"[Proximity Search] Enabled with TTL={proximity_ttl}, "
                  f"Expand={proximity_expand}x, Thresh={_proximity_thresh:.2f}")

    for frame in read_iter:
        # Perform network inference, get bb dets and landmarks
        dets, lms = centerface(frame, threshold=threshold)

        # Apply proximity search if enabled (reacquire missed faces via ROI detection)
        if proximity_search_mgr is not None:
            # Update confirmed face states from fresh detections
            proximity_search_mgr.update_confirmed_detections(dets, lms, frame_idx)

            # Attempt to reacquire missing confirmed faces
            reacquired_dets, reacquired_lms = proximity_search_mgr.search_missing_faces(
                frame, dets, frame_idx, threshold
            )

            # Merge reacquired detections with original
            if len(reacquired_dets) > 0:
                dets = np.vstack([dets, reacquired_dets])
                lms = np.vstack([lms, reacquired_lms])

        # Track original detection count before gap-filling augmentation
        original_dets_count = len(dets)

        # Apply tracking if enabled (injects gap-filled detections)
        if tracker is not None:
            dets, lms = tracker.update(dets, lms, frame_idx)

        anonymize_frame(
            dets, frame, mask_scale=mask_scale,
            replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
            replaceimg=replaceimg, mosaicsize=mosaicsize,
            original_dets_count=original_dets_count if tracker is not None else None
        )

        if opath is not None:
            writer.append_data(frame)

        if enable_preview:
            cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])  # RGB -> RGB
            if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:  # 27 is the escape key code
                cv2.destroyAllWindows()
                break
        bar.update()
        frame_idx += 1

    reader.close()
    if opath is not None:
        writer.close()
    bar.close()

    # Print proximity search statistics if enabled
    if proximity_search_mgr is not None and not disable_progress_output:
        stats = proximity_search_mgr.get_stats()
        print(f"[Proximity Search] Stats: {stats['total_roi_searches']} ROI searches, "
              f"{stats['successful_reacquisitions']} successful reacquisitions, "
              f"{stats['failed_validations']} failed validations")

    # Print tracking statistics if enabled
    if tracker is not None and not disable_progress_output:
        stats = tracker.get_stats()
        print(f"[Tracking] Stats: {stats['total_detections']} detections, "
              f"{stats['total_tracks_created']} tracks created, "
              f"{stats['frames_with_gaps_filled']} frames gap-filled, "
              f"{stats['total_gap_fills']} total gap-fills")


def image_detect(
        ipath: str,
        opath: str,
        centerface: Any,
        threshold: float,
        replacewith: str,
        mask_scale: float,
        ellipse: bool,
        draw_scores: bool,
        enable_preview: bool,
        keep_metadata: bool,
        replaceimg = None,
        mosaicsize: int = 20,
):
    frame = iio.imread(ipath)

    if keep_metadata:
        # Source image EXIF metadata retrieval via imageio V3 lib
        metadata = imageio.v3.immeta(ipath)
        exif_dict = metadata.get("exif", None)

    # Perform network inference, get bb dets but discard landmark predictions
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg, mosaicsize=mosaicsize
    )

    if enable_preview:
        cv2.imshow('Preview of anonymization results (quit by pressing Q or Escape)', frame[:, :, ::-1])  # RGB -> RGB
        if cv2.waitKey(0) & 0xFF in [ord('q'), 27]:  # 27 is the escape key code
            cv2.destroyAllWindows()

    imageio.imsave(opath, frame)

    if keep_metadata:
        # Save image with EXIF metadata
        imageio.imsave(opath, frame, exif=exif_dict)

    # print(f'Output saved to {opath}')


def get_file_type(path):
    if path.startswith('<video'):
        return 'cam'
    if not os.path.isfile(path):
        return 'notfound'
    mime = mimetypes.guess_type(path)[0]
    if mime is None:
        return None
    if mime.startswith('video'):
        return 'video'
    if mime.startswith('image'):
        return 'image'
    return mime


def get_anonymized_image(frame,
                         threshold: float,
                         replacewith: str,
                         mask_scale: float,
                         ellipse: bool,
                         draw_scores: bool,
                         replaceimg = None
                         ):
    """
    Method for getting an anonymized image without CLI
    returns frame
    """

    from deface.centerface import CenterFace

    centerface = CenterFace(in_shape=None, backend='auto')
    dets, _ = centerface(frame, threshold=threshold)

    anonymize_frame(
        dets, frame, mask_scale=mask_scale,
        replacewith=replacewith, ellipse=ellipse, draw_scores=draw_scores,
        replaceimg=replaceimg
    )

    return frame


def parse_cli_args():
    parser = argparse.ArgumentParser(description='Video anonymization by face detection', add_help=False)
    parser.add_argument(
        'input', nargs='*',
        help=f'File path(s) or camera device name. It is possible to pass multiple paths by separating them by spaces or by using shell expansion (e.g. `$ deface vids/*.mp4`). Alternatively, you can pass a directory as an input, in which case all files in the directory will be used as inputs. If a camera is installed, a live webcam demo can be started by running `$ deface cam` (which is a shortcut for `$ deface -p \'<video0>\'`.')
    parser.add_argument(
        '--output', '-o', default=None, metavar='O',
        help='Output file name. Defaults to input path + postfix "_anonymized".')
    parser.add_argument(
        '--detector', default='scrfd2.5g', choices=['scrfd2.5g','scrfd10g', 'centerface', 'yolo'],
        help='Detector backend. Default: "scrfd2.5g". Use "yolo" for person detection.')
    parser.add_argument(
        '--thresh', '-t', default=0.3, type=float, metavar='T',
        help='Detection threshold (tune this to trade off between false positive and false negative rate). Default: 0.3.')
    parser.add_argument(
        '--scale', '-s', default=None, metavar='WxH',
        help='Downscale images for network inference to this size (format: WxH, example: --scale 640x360).')
    parser.add_argument(
        '--preview', '-p', default=False, action='store_true',
        help='Enable live preview GUI (can decrease performance).')
    parser.add_argument(
        '--boxes', default=False, action='store_true',
        help='Use boxes instead of ellipse masks.')
    parser.add_argument(
        '--draw-scores', default=False, action='store_true',
        help='Draw detection scores onto outputs.')
    parser.add_argument(
        '--scores', default=False, action='store_true',
        help='Debug mode: draw bounding boxes with confidence scores instead of blurring faces.')
    parser.add_argument(
        '--disable-progress-output', default=False, action='store_true',
        help='Disable video progress output to console.')
    parser.add_argument(
        '--mask-scale', default=1.3, type=float, metavar='M',
        help='Scale factor for face masks, to make sure that masks cover the complete face. Default: 1.3.')
    parser.add_argument(
        '--replacewith', default='blur', choices=['blur', 'solid', 'none', 'img', 'mosaic'],
        help='Anonymization filter mode for face regions. "blur" applies a strong gaussian blurring, "solid" draws a solid black box, "none" does leaves the input unchanged, "img" replaces the face with a custom image and "mosaic" replaces the face with mosaic. Default: "blur".')
    parser.add_argument(
        '--replaceimg', default='replace_img.png',
        help='Anonymization image for face regions. Requires --replacewith img option.')
    parser.add_argument(
        '--mosaicsize', default=20, type=int, metavar='width',
        help='Setting the mosaic size. Requires --replacewith mosaic option. Default: 20.')
    parser.add_argument(
        '--yolo-variant', default='v4', choices=['v4', 'v8'], metavar='VARIANT',
        help='YOLO model variant when using --detector yolo. "v4" uses yolov4.onnx, "v8" uses yolov8.onnx. Default: "v4".')
    parser.add_argument(
        '--enable-tracking', default=False, action='store_true',
        help='Enable face tracking to fill detection gaps (1-10 frames). Disabled by default.')
    parser.add_argument(
        '--track-iou', default=0.25, type=float, metavar='IOU',
        help='IoU threshold for track-to-detection association. Default: 0.25.')
    parser.add_argument(
        '--track-distance', default=1.5, type=float, metavar='DIST',
        help='Normalized distance threshold for track association (multiplier of box diagonal). Default: 1.5.')
    parser.add_argument(
        '--track-alpha', default=0.65, type=float, metavar='ALPHA',
        help='EMA smoothing factor for track box updates (0-1, higher = more responsive). Default: 0.65.')
    parser.add_argument(
        '--track-ttl', default=10, type=int, metavar='TTL',
        help='Track time-to-live: continue blurring for this many frames after detection loss, then delete. Default: 10.')
    parser.add_argument(
        '--track-expansion', default=0.05, type=float, metavar='EXPANSION',
        help='Box expansion factor per miss during gap-filling (5% = 0.05 per side). Default: 0.05.')
    parser.add_argument(
        '--track-debug', default=False, action='store_true',
        help='Enable debug output for tracking (prints matches and gap-fills per frame).')
    parser.add_argument(
        '--enable-proximity-search', default=False, action='store_true',
        help='Enable proximity search to reacquire missed faces via ROI-based detection. Default: disabled.')
    parser.add_argument(
        '--proximity-ttl', default=10, type=int, metavar='TTL',
        help='Max frames since last confirmation to attempt proximity search. Default: 10.')
    parser.add_argument(
        '--proximity-expand', default=1.8, type=float, metavar='FACTOR',
        help='ROI expansion factor around last confirmed box. Default: 1.8.')
    parser.add_argument(
        '--proximity-thresh', default=None, type=float, metavar='THRESH',
        help='Detection threshold for proximity ROIs (None = 0.8 * main threshold). Default: None.')
    parser.add_argument(
        '--proximity-iou', default=0.15, type=float, metavar='IOU',
        help='Minimum IoU with last confirmed box for validation. Default: 0.15.')
    parser.add_argument(
        '--proximity-dist', default=1.2, type=float, metavar='DIST',
        help='Max normalized center distance for validation (× box diagonal). Default: 1.2.')
    parser.add_argument(
        '--proximity-area-min', default=0.4, type=float, metavar='MIN',
        help='Min area ratio vs last confirmed box (reject too-small candidates). Default: 0.4.')
    parser.add_argument(
        '--proximity-area-max', default=2.5, type=float, metavar='MAX',
        help='Max area ratio vs last confirmed box (reject too-large candidates). Default: 2.5.')
    parser.add_argument(
        '--proximity-debug', default=False, action='store_true',
        help='Enable debug output for proximity search (prints per-frame stats).')
    parser.add_argument(
        '--keep-audio', '-k', default=False, action='store_true',
        help='Keep audio from video source file and copy it over to the output (only applies to videos).')
    parser.add_argument(
        '--ffmpeg-config', default={"codec": "libx264"}, type=json.loads,
        help='FFMPEG config arguments for encoding output videos. This argument is expected in JSON notation. For a list of possible options, refer to the ffmpeg-imageio docs. Default: \'{"codec": "libx264"}\'.'
    )  # See https://imageio.readthedocs.io/en/stable/format_ffmpeg.html#parameters-for-saving
    parser.add_argument(
        '--backend', default='auto', choices=['auto', 'onnxrt', 'opencv'],
        help='Backend for ONNX model execution. Default: "auto" (prefer onnxrt if available).')
    parser.add_argument(
        '--execution-provider', '--ep', default=None, metavar='EP',
        help='Override onnxrt execution provider (see https://onnxruntime.ai/docs/execution-providers/). If not specified, the presumably fastest available one will be automatically selected. Only used if backend is onnxrt.')
    parser.add_argument(
        '--version', action='version', version=__version__,
        help='Print version number and exit.')
    parser.add_argument(
        '--keep-metadata', '-m', default=False, action='store_true',
        help='Keep metadata of the original image. Default : False.')
    parser.add_argument('--help', '-h', action='help', help='Show this help message and exit.')

    args = parser.parse_args()

    if len(args.input) == 0:
        parser.print_help()
        print('\nPlease supply at least one input path.')
        exit(1)

    if args.input == ['cam']:  # Shortcut for webcam demo with live preview
        args.input = ['<video0>']
        args.preview = True

    return args


def main():
    args = parse_cli_args()
    ipaths = []

    # add files in folders
    for path in args.input:
        if os.path.isdir(path):
            for file in os.listdir(path):
                ipaths.append(os.path.join(path,file))
        else:
            # Either a path to a regular file, the special 'cam' shortcut
            # or an invalid path. The latter two cases are handled below.
            ipaths.append(path)

    #comfy variables
    base_opath = args.output
    replacewith = args.replacewith
    enable_preview = args.preview
    draw_scores = args.draw_scores
    threshold = args.thresh
    ellipse = not args.boxes
    mask_scale = args.mask_scale
    keep_audio = args.keep_audio
    ffmpeg_config = args.ffmpeg_config
    backend = args.backend
    in_shape = args.scale
    execution_provider = args.execution_provider
    mosaicsize = args.mosaicsize
    keep_metadata = args.keep_metadata
    replaceimg = None
    disable_progress_output = args.disable_progress_output

    # Tracking parameters
    enable_tracking = args.enable_tracking
    track_iou_threshold = args.track_iou
    track_dist_threshold = args.track_distance
    track_alpha = args.track_alpha
    track_ttl = args.track_ttl
    track_expansion = args.track_expansion
    track_debug = args.track_debug

    # Proximity search parameters
    enable_proximity_search = args.enable_proximity_search
    proximity_ttl = args.proximity_ttl
    proximity_expand = args.proximity_expand
    proximity_thresh = args.proximity_thresh
    proximity_iou = args.proximity_iou
    proximity_dist = args.proximity_dist
    proximity_area_min = args.proximity_area_min
    proximity_area_max = args.proximity_area_max
    proximity_debug = args.proximity_debug

    # When --scores flag is used, override to draw boxes with confidence scores instead of blurring
    if args.scores:
        replacewith = 'none'
        draw_scores = True

    if in_shape is not None:
        w, h = in_shape.split('x')
        in_shape = int(w), int(h)
    if replacewith == "img":
        replaceimg = imageio.imread(args.replaceimg)
        print(f'After opening {args.replaceimg} shape: {replaceimg.shape}')


    # TODO: scalar downscaling setting (-> in_shape), preserving aspect ratio
    if args.detector == 'scrfd2.5g':
        from deface.scrfd_detector import SCRFDdetector

        if not os.path.isfile(default_scrfd_onnx_path):
            raise RuntimeError(
                f'SCRFD detector selected but default model file not found at {default_scrfd_onnx_path}. '
                'Provide the model file there or use --detector centerface.'
            )
        detector = SCRFDdetector(
            model_path=default_scrfd_onnx_path,
            device='auto',
            override_execution_provider=execution_provider,
        )
    elif args.detector == 'scrfd10g':
        from deface.scrfd10g_detector import SCRFD10GDetector

        scrfd_10g_onnx_path = os.path.join(_models_dir, 'scrfd_10g.onnx')
        if not os.path.isfile(scrfd_10g_onnx_path):
            raise RuntimeError(
                f'SCRFD 10G detector selected but model file not found at {scrfd_10g_onnx_path}. '
                'Provide the model file there or use --detector centerface.'
            )
        detector = SCRFD10GDetector(
            model_path=scrfd_10g_onnx_path,
            device='auto',
            override_execution_provider=execution_provider,
        )
    elif args.detector == 'yolo':
        from deface.yolo_detector import YOLODetector

        # Determine which YOLO model to use based on variant
        yolo_variant = args.yolo_variant.lower()
        if yolo_variant == 'v8':
            yolo_onnx_path = os.path.join(_models_dir, 'yolov8.onnx')
            model_name = 'yolov8.onnx'
        else:
            yolo_onnx_path = os.path.join(_models_dir, 'yolov4.onnx')
            model_name = 'yolov4.onnx'

        if not os.path.isfile(yolo_onnx_path):
            raise RuntimeError(
                f'YOLO detector selected but model file not found at {yolo_onnx_path}. '
                'Provide the model file there or use --detector centerface.'
            )
        detector = YOLODetector(
            model_path=yolo_onnx_path,
            device='auto',
            override_execution_provider=execution_provider,
            variant=yolo_variant,
        )
    elif args.detector == 'centerface':
        from deface.centerface import CenterFace

        detector = CenterFace(
            in_shape=in_shape,
            backend=backend,
            override_execution_provider=execution_provider,
        )
    else:
        raise RuntimeError(f'Unknown detector: {args.detector}')

    multi_file = len(ipaths) > 1
    if multi_file:
        ipaths = tqdm.tqdm(ipaths, position=0, dynamic_ncols=True, desc='Batch progress')

    for ipath in ipaths:
        opath = base_opath
        if ipath == 'cam':
            ipath = '<video0>'
            enable_preview = True
        filetype = get_file_type(ipath)
        is_cam = filetype == 'cam'
        if opath is None and not is_cam:
            root, ext = os.path.splitext(ipath)
            opath = f'{root}_anonymized{ext}'
        print(f'Input:  {ipath}\nOutput: {opath}')
        if opath is None and not enable_preview:
            print('No output file is specified and the preview GUI is disabled. No output will be produced.')
        if filetype == 'video' or is_cam:
            video_detect(
                ipath=ipath,
                opath=opath,
                centerface=detector,
                threshold=threshold,
                cam=is_cam,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                nested=multi_file,
                keep_audio=keep_audio,
                ffmpeg_config=ffmpeg_config,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize,
                disable_progress_output=disable_progress_output,
                enable_tracking=enable_tracking,
                track_iou_threshold=track_iou_threshold,
                track_dist_threshold=track_dist_threshold,
                track_alpha=track_alpha,
                track_ttl=track_ttl,
                track_expansion=track_expansion,
                track_debug=track_debug,
                enable_proximity_search=enable_proximity_search,
                proximity_ttl=proximity_ttl,
                proximity_expand=proximity_expand,
                proximity_thresh=proximity_thresh,
                proximity_iou=proximity_iou,
                proximity_dist=proximity_dist,
                proximity_area_min=proximity_area_min,
                proximity_area_max=proximity_area_max,
                proximity_debug=proximity_debug
            )
        elif filetype == 'image':
            image_detect(
                ipath=ipath,
                opath=opath,
                centerface=detector,
                threshold=threshold,
                replacewith=replacewith,
                mask_scale=mask_scale,
                ellipse=ellipse,
                draw_scores=draw_scores,
                enable_preview=enable_preview,
                keep_metadata=keep_metadata,
                replaceimg=replaceimg,
                mosaicsize=mosaicsize
            )
        elif filetype is None:
            print(f'Can\'t determine file type of file {ipath}. Skipping...')
        elif filetype == 'notfound':
            print(f'File {ipath} not found. Skipping...')
        else:
            print(f'File {ipath} has an unknown type {filetype}. Skipping...')


if __name__ == '__main__':
    main()
