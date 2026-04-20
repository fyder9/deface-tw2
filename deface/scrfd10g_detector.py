import cv2
import numpy as np


class SCRFD10GDetector:
    """
    Dedicated detector for SCRFD 10G model with robust output handling.
    Designed to handle the specific output array shapes of SCRFD 10G model.
    Includes comprehensive debug logging to diagnose array shape mismatches.
    """

    def __init__(self, model_path: str, device: str = "cpu", cap_long_side: int = 1920, override_execution_provider: str = None):
        import onnxruntime

        self.cap_long_side = cap_long_side
        self.debug = False  # Set to True for debugging array shapes and decoding

        available = onnxruntime.get_available_providers()

        def _make_session(providers_list):
            return onnxruntime.InferenceSession(model_path, providers=providers_list)

        if override_execution_provider is not None:
            if override_execution_provider not in available:
                raise ValueError(
                    f"Requested execution provider '{override_execution_provider}' is not available. "
                    f"Available providers: {available}"
                )
            candidates = [[override_execution_provider, "CPUExecutionProvider"]]
        else:
            dev = (device or "cpu").lower()
            if dev == "cpu":
                candidates = [["CPUExecutionProvider"]]
            else:
                cand = []
                if "CUDAExecutionProvider" in available:
                    cand.append(["CUDAExecutionProvider", "CPUExecutionProvider"])
                if "DmlExecutionProvider" in available:
                    cand.append(["DmlExecutionProvider", "CPUExecutionProvider"])
                cand.append(["CPUExecutionProvider"])
                candidates = cand

        last_err = None
        self.sess = None
        for provs in candidates:
            try:
                self.sess = _make_session(provs)
                break
            except Exception as e:
                last_err = e
                continue

        if self.sess is None:
            raise RuntimeError(f"Failed to create ONNX Runtime session. Last error: {last_err}")

        preferred_provider = self.sess.get_providers()[0] if self.sess.get_providers() else "<none>"
        print("[SCRFD10G] Available providers:", available)
        print("[SCRFD10G] Selected providers:", self.sess.get_providers())
        print("[SCRFD10G] Running on:", preferred_provider)

        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [output.name for output in self.sess.get_outputs()]
        self.nms_iou = 0.4
        self.score_thresh = 0.2

    def ensure_rgb(self, img: np.ndarray) -> np.ndarray:
        """Convert grayscale or RGBA to RGB"""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img

    def resize_frame(self, frame: np.ndarray) -> tuple[np.ndarray, int, int, float, int, int]:
        """
        Resize frame maintaining aspect ratio and apply padding to align stride=32.

        Returns:
            (padded_frame, padded_w, padded_h, scale, org_w, org_h)
            - padded_frame: Frame with padding applied
            - padded_w, padded_h: Dimensions after padding (multiples of 32)
            - scale: Scale factor from original to resized
            - org_w, org_h: Original frame dimensions
        """
        h, w = frame.shape[:2]
        org_w, org_h = w, h
        longside = max(h, w)

        # Resize if needed
        if longside > self.cap_long_side:
            scale = self.cap_long_side / longside
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        else:
            scale = 1.0
            new_w, new_h = w, h

        # Align to stride=32 for SCRFD 10G FPN compatibility
        stride = 32
        padded_w = ((new_w + stride - 1) // stride) * stride
        padded_h = ((new_h + stride - 1) // stride) * stride

        # Apply padding only if necessary (bottom/right padding with black border)
        if padded_w != new_w or padded_h != new_h:
            pad_right = padded_w - new_w
            pad_bottom = padded_h - new_h
            frame = cv2.copyMakeBorder(
                frame, 0, pad_bottom, 0, pad_right,
                cv2.BORDER_CONSTANT, value=0
            )

        if self.debug:
            print(f"[SCRFD10G] resize_frame: org({org_w}x{org_h}) -> resized({new_w}x{new_h}, scale={scale:.4f}) -> padded({padded_w}x{padded_h})")

        return frame, padded_w, padded_h, scale, org_w, org_h

    def preprocess(self, img_rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to BGR, normalize, and prepare blob for inference"""
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        img_bgr = (img_bgr - 127.5) / 128.0
        img_bgr = np.transpose(img_bgr, (2, 0, 1))  # HWC to CHW
        img_bgr = np.expand_dims(img_bgr, axis=0)  # Add batch dimension
        return img_bgr

    def infer(self, frame: np.ndarray):
        """Perform inference and return raw outputs with metadata"""
        frame = self.ensure_rgb(frame)
        frame, padded_w, padded_h, scale, org_w, org_h = self.resize_frame(frame)
        blob = self.preprocess(frame)

        if self.debug:
            print(f"[SCRFD10G] infer: blob shape = {blob.shape}")

        outputs = self.sess.run(self.output_names, {self.input_name: blob})

        if self.debug:
            print(f"[SCRFD10G] infer: num outputs = {len(outputs)}")
            for i, out in enumerate(outputs):
                print(f"[SCRFD10G]   output[{i}]: shape={out.shape}, dtype={out.dtype}")

        return outputs, padded_w, padded_h, scale, org_w, org_h

    def _safe_float_conversion(self, value):
        """Safely convert array value to Python float"""
        # Handle numpy scalars and arrays
        if isinstance(value, np.ndarray):
            if value.ndim == 0:  # 0-d array
                return float(value.item())
            elif value.size == 1:  # Array with single element
                return float(value.flat[0])
            else:
                raise ValueError(f"Cannot convert multi-element array to scalar: shape={value.shape}")
        else:
            return float(value)

    def decode_outputs(self, outputs, padded_w, padded_h, scale, org_w, org_h, score_thresh: float | None = None):
        """
        Decode raw ONNX outputs to pixel coordinates and landmarks.
        Robust handling of SCRFD 10G output array structures.

        NOTE: SCRFD 10G has different output ordering than SCRFD 2.5G:
        - SCRFD 2.5G: [scores0, scores1, scores2, bboxes0, bboxes1, bboxes2, kps0, kps1, kps2]
        - SCRFD 10G:  [bboxes0, scores0, kps0, bboxes1, scores1, kps1, bboxes2, scores2, kps2]

        Args:
            outputs: List of 9 arrays from SCRFD10G (3 scales × 3 outputs: bboxes, scores, landmarks)
            padded_w, padded_h: Dimensions after padding
            scale: Scale factor from original to resized
            org_w, org_h: Original dimensions
            score_thresh: Score threshold for detection

        Returns:
            (pixel_boxes, pixel_lms) where each box is [x1, y1, x2, y2, score]
        """
        pixel_boxes = []
        pixel_lms = []

        # Process 3 pyramid levels (stride 8, 16, 32)
        # SCRFD 10G output order: [bbox_s8, score_s8, kps_s8, bbox_s16, score_s16, kps_s16, bbox_s32, score_s32, kps_s32]
        for scale_idx in range(3):
            bboxes = outputs[scale_idx * 3]      # Bboxes for this scale
            scores = outputs[scale_idx * 3 + 1]  # Scores for this scale
            kps = outputs[scale_idx * 3 + 2]     # Landmarks for this scale
            stride = 8 * (2 ** scale_idx)

            if self.debug:
                print(f"[SCRFD10G] scale_idx={scale_idx}, stride={stride}")
                print(f"[SCRFD10G]   scores shape={scores.shape}, dtype={scores.dtype}")
                print(f"[SCRFD10G]   bboxes shape={bboxes.shape}, dtype={bboxes.dtype}")
                print(f"[SCRFD10G]   kps shape={kps.shape}, dtype={kps.dtype}")

            # Calculate feature map dimensions from batch/anchor dimension
            # SCRFD 10G: outputs have shape (batch=1, num_anchors, values)
            # Extract first batch element
            scores_batch = scores[0]  # Shape: (num_anchors, 1) or (num_anchors, score_channels)
            bboxes_batch = bboxes[0]  # Shape: (num_anchors, 4)
            kps_batch = kps[0]        # Shape: (num_anchors, 10)

            num_anchors = scores_batch.shape[0]
            fm_h = int(np.ceil(padded_h / stride))
            fm_w = int(np.ceil(padded_w / stride))

            if self.debug:
                print(f"[SCRFD10G]   num_anchors: {num_anchors}, feature map: {fm_h}x{fm_w}")

            # Number of anchors per location
            A = num_anchors // (fm_w * fm_h) if fm_w * fm_h > 0 else 1

            if self.debug:
                print(f"[SCRFD10G]   anchors per location: {A}")

            # Iterate through all anchors
            for i in range(num_anchors):
                try:
                    # Extract and convert score safely
                    # scores_batch[i] can be shape (1,) or scalar
                    score_val = scores_batch[i]
                    score = self._safe_float_conversion(score_val)

                except (ValueError, IndexError, TypeError) as e:
                    if self.debug and i < 5:  # Log first 5 errors
                        print(f"[SCRFD10G] Error converting score at index {i}: {e}")
                    continue

                thresh = self.score_thresh if score_thresh is None else float(score_thresh)
                if score <= thresh:
                    continue

                # Calculate grid position
                cell = i // A
                gx = cell % fm_w
                gy = cell // fm_w

                # Center position in padded frame
                px = (gx + 0.5) * stride
                py = (gy + 0.5) * stride

                # Extract bbox offsets from bboxes_batch[i]
                bbox_vals = bboxes_batch[i]  # Shape: (4,)
                dx1 = self._safe_float_conversion(bbox_vals[0]) if bbox_vals.size > 0 else 0.0
                dy1 = self._safe_float_conversion(bbox_vals[1]) if bbox_vals.size > 1 else 0.0
                dx2 = self._safe_float_conversion(bbox_vals[2]) if bbox_vals.size > 2 else 0.0
                dy2 = self._safe_float_conversion(bbox_vals[3]) if bbox_vals.size > 3 else 0.0

                # Convert to original coordinate space
                x1 = (px - dx1 * stride) / scale
                y1 = (py - dy1 * stride) / scale
                x2 = (px + dx2 * stride) / scale
                y2 = (py + dy2 * stride) / scale

                # Clip to original frame boundaries
                x1 = max(0, min(x1, org_w - 1))
                y1 = max(0, min(y1, org_h - 1))
                x2 = max(0, min(x2, org_w - 1))
                y2 = max(0, min(y2, org_h - 1))

                # Skip invalid boxes
                if x2 <= x1 or y2 <= y1:
                    continue

                pixel_boxes.append([x1, y1, x2, y2, score])

                # Extract landmarks (5 points × 2 coords) from kps_batch[i]
                lm_out = []
                kps_vals = kps_batch[i]  # Shape: (10,)
                for j in range(5):
                    try:
                        lx_offset = self._safe_float_conversion(kps_vals[2 * j]) if kps_vals.size > 2 * j else 0.0
                        ly_offset = self._safe_float_conversion(kps_vals[2 * j + 1]) if kps_vals.size > 2 * j + 1 else 0.0
                        lx = (px + lx_offset * stride) / scale
                        ly = (py + ly_offset * stride) / scale
                        lm_out.extend([lx, ly])
                    except (ValueError, IndexError) as e:
                        if self.debug and i < 5:
                            print(f"[SCRFD10G] Error converting landmark at index {i}, point {j}: {e}")
                        lm_out.extend([0.0, 0.0])

                pixel_lms.append(lm_out)

        if self.debug:
            print(f"[SCRFD10G] decode_outputs: found {len(pixel_boxes)} boxes")

        return pixel_boxes, pixel_lms

    def nms(self, pixel_boxes, pixel_lms, iou_threshold=0.4):
        """Apply Non-Maximum Suppression using OpenCV"""
        if not pixel_boxes:
            return [], []

        bboxes = []
        scores = []
        for (x1, y1, x2, y2, s) in pixel_boxes:
            bboxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])  # xywh
            scores.append(float(s))

        idx = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
        idx = idx.flatten().tolist() if len(idx) else []

        kept_boxes = [pixel_boxes[i] for i in idx]
        kept_lms = [pixel_lms[i] for i in idx] if pixel_lms is not None else None

        if self.debug:
            print(f"[SCRFD10G] nms: {len(pixel_boxes)} boxes -> {len(kept_boxes)} after NMS")

        return kept_boxes, kept_lms

    def draw_blurred_boxes(self, frame: np.ndarray, boxes: list):
        """Apply Gaussian blur to detected face regions"""
        for (x1, y1, x2, y2, s) in boxes:
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            face_roi = frame[y1:y2, x1:x2]
            if face_roi.size == 0:
                continue
            ksize = (max(1, (x2 - x1) // 7 | 1), max(1, (y2 - y1) // 7 | 1))
            blurred_face = cv2.GaussianBlur(face_roi, ksize, 0)
            frame[y1:y2, x1:x2] = blurred_face
        return frame

    def __call__(self, frame: np.ndarray, threshold: float = 0.5):
        """
        CenterFace-compatible callable interface.
        Returns detections and landmarks in the same format as other detectors.
        """
        outputs, padded_w, padded_h, scale, org_w, org_h = self.infer(frame)
        pixel_boxes, pixel_lms = self.decode_outputs(outputs, padded_w, padded_h, scale, org_w, org_h, score_thresh=threshold)
        kept_boxes, kept_lms = self.nms(pixel_boxes, pixel_lms, iou_threshold=self.nms_iou)

        # Return in standard format: (Nx5 array of boxes, Nx10 array of landmarks)
        dets = np.asarray(kept_boxes, dtype=np.float32) if kept_boxes else np.zeros((0, 5), np.float32)
        lms = np.asarray(kept_lms, dtype=np.float32) if kept_lms else np.zeros((0, 10), np.float32)

        return dets, lms
