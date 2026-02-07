import cv2
import numpy as np
import math


class YOLODetector:
    """
    YOLO-based person detector with ONNX Runtime GPU acceleration.
    Designed for detecting people in frames with high precision.

    Supports both YOLOv4 and YOLOv8 models.

    YOLOv4 Input spec:
    - Shape: (1, 416, 416, 3)
    - Format: RGB, float32, normalized [0, 1]
    - Letterboxed with constant padding (no stretching)
    - NHWC format

    YOLOv4 Output spec:
    - 3 output layers with shape (1, grid, grid, 3, 85)
    - 85 values per anchor: [x, y, w, h, objectness, class_probs(80)]

    YOLOv8 Input spec:
    - Shape: (1, 3, 640, 640)
    - Format: RGB, float32, normalized [0, 1]
    - Letterboxed with constant padding
    - NCHW format

    YOLOv8 Output spec:
    - Single output layer with shape (1, 5, num_predictions)
    - 5 values per prediction: [x, y, w, h, objectness]
    """

    def __init__(self, model_path: str, device: str = "cpu", override_execution_provider: str = None, variant: str = "v4"):
        import onnxruntime

        self.variant = variant.lower()
        if self.variant not in ["v4", "v8"]:
            raise ValueError(f"Unsupported YOLO variant: {variant}. Use 'v4' or 'v8'.")

        # Set model-specific parameters
        if self.variant == "v4":
            self.input_size = 416
            self.anchors = [
                [(10, 13), (16, 30), (33, 23)],      # stride 32
                [(30, 61), (62, 45), (59, 119)],     # stride 16
                [(116, 90), (156, 198), (373, 326)]  # stride 8
            ]
            self.strides = [32, 16, 8]
        else:  # v8
            self.input_size = 640
            # YOLOv8 doesn't use explicit anchors; decode differently
            self.anchors = None
            self.strides = [8, 16, 32]

        self.num_classes = 80
        self.xy_scale = 1.05  # xy scale factor in YOLOv4 decoding

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
        print("[YOLO] Available providers:", available)
        print("[YOLO] Selected providers:", self.sess.get_providers())
        print("[YOLO] Running on:", preferred_provider)

        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [output.name for output in self.sess.get_outputs()]
        self.nms_iou = 0.45
        self.score_thresh = 0.3
        # Only person class (COCO class 0)
        self.target_class_id = 0

    def ensure_rgb(self, img: np.ndarray) -> np.ndarray:
        """Convert grayscale or RGBA to RGB"""
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        elif img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img

    def preprocess(self, frame: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        """
        Preprocess image for YOLO inference with letterboxing.

        YOLOv4: Returns NHWC format (batch, H, W, 3)
        YOLOv8: Returns NCHW format (batch, 3, H, W)

        Returns:
            (input_blob, resize_ratio, pad_w, pad_h)
        """
        h, w = frame.shape[:2]

        # Compute resize scale to fit input_size while maintaining aspect ratio
        resize_ratio = min(self.input_size / w, self.input_size / h)
        new_w = int(w * resize_ratio)
        new_h = int(h * resize_ratio)

        # Resize keeping aspect ratio
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Create letterbox: pad to input_size with constant value 128
        pad_w = (self.input_size - new_w) // 2
        pad_h = (self.input_size - new_h) // 2

        letterboxed = np.full((self.input_size, self.input_size, 3), 128, dtype=np.uint8)
        letterboxed[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert BGR -> RGB (OpenCV reads as BGR)
        letterboxed_rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB).astype(np.float32)

        # Normalize to [0, 1]
        letterboxed_rgb = letterboxed_rgb / 255.0

        # Format depends on model version
        if self.variant == "v4":
            # YOLOv4: NHWC format (batch, height, width, channels)
            input_blob = np.expand_dims(letterboxed_rgb, axis=0).astype(np.float32)
        else:
            # YOLOv8: NCHW format (batch, channels, height, width)
            # Transpose from HWC to CHW, then add batch dimension
            input_blob = np.transpose(letterboxed_rgb, (2, 0, 1))
            input_blob = np.expand_dims(input_blob, axis=0).astype(np.float32)

        return input_blob, resize_ratio, pad_w, pad_h

    def infer(self, frame: np.ndarray) -> tuple:
        """
        Perform inference and return raw outputs with metadata.

        Returns:
            (outputs, h, w, resize_ratio, pad_w, pad_h)
        """
        frame = self.ensure_rgb(frame)
        h, w = frame.shape[:2]

        input_blob, resize_ratio, pad_w, pad_h = self.preprocess(frame)
        outputs = self.sess.run(self.output_names, {self.input_name: input_blob})

        return outputs, h, w, resize_ratio, pad_w, pad_h

    def _sigmoid(self, x):
        """Sigmoid activation"""
        return 1.0 / (1.0 + np.exp(-x))

    def _decode_predictions_v8(self, output, org_h, org_w, resize_ratio, pad_w, pad_h):
        """
        Decode YOLOv8 output format.

        YOLOv8 output shape: (batch, 5, num_predictions)
        5 values: [x, y, w, h, objectness]

        Returns:
            List of [x1, y1, x2, y2, objectness] in original image coordinates
        """
        pixel_boxes = []

        # Output shape: (1, 5, num_predictions)
        batch_output = output[0]  # Shape: (5, num_predictions)
        num_predictions = batch_output.shape[1]

        for pred_idx in range(num_predictions):
            x = float(batch_output[0, pred_idx])
            y = float(batch_output[1, pred_idx])
            w = float(batch_output[2, pred_idx])
            h = float(batch_output[3, pred_idx])
            objectness = float(batch_output[4, pred_idx])

            # Convert center coordinates and dimensions to box
            x1_letterbox = x - w / 2
            y1_letterbox = y - h / 2
            x2_letterbox = x + w / 2
            y2_letterbox = y + h / 2

            # Reverse letterbox padding
            x1 = (x1_letterbox - pad_w) / resize_ratio
            y1 = (y1_letterbox - pad_h) / resize_ratio
            x2 = (x2_letterbox - pad_w) / resize_ratio
            y2 = (y2_letterbox - pad_h) / resize_ratio

            # Clip to original image bounds
            x1 = max(0, min(x1, org_w - 1))
            y1 = max(0, min(y1, org_h - 1))
            x2 = max(0, min(x2, org_w - 1))
            y2 = max(0, min(y2, org_h - 1))

            # Skip invalid boxes
            if x2 <= x1 or y2 <= y1:
                continue

            pixel_boxes.append([x1, y1, x2, y2, objectness])

        return pixel_boxes

    def _decode_predictions(self, raw_pred, stride, anchors_for_stride):
        """
        Decode raw predictions from a single YOLO output layer.

        Args:
            raw_pred: Shape (1, grid_h, grid_w, 3, 85) - raw model output
            stride: Stride for this layer (32, 16, or 8)
            anchors_for_stride: List of 3 anchor tuples for this stride

        Returns:
            List of [x, y, w, h, objectness, class_probs]
        """
        predictions = []

        # Extract batch
        batch_pred = raw_pred[0]  # Shape: (grid_h, grid_w, 3, 85)
        grid_h, grid_w, num_anchors, pred_size = batch_pred.shape

        for gy in range(grid_h):
            for gx in range(grid_w):
                for anchor_idx in range(num_anchors):
                    anchor_w, anchor_h = anchors_for_stride[anchor_idx]
                    pred = batch_pred[gy, gx, anchor_idx]  # Shape: (85,)

                    # Extract components
                    dx = float(pred[0])
                    dy = float(pred[1])
                    dw = float(pred[2])
                    dh = float(pred[3])
                    objectness = float(pred[4])
                    class_probs = pred[5:5+self.num_classes].astype(np.float32)

                    # Decode bbox with YOLO formula
                    # pred_xy = ((sigmoid(dxdy) * xyscale) - 0.5*(xyscale-1) + grid_xy) * stride
                    sig_dx = self._sigmoid(dx)
                    sig_dy = self._sigmoid(dy)

                    pred_x = ((sig_dx * self.xy_scale) - 0.5 * (self.xy_scale - 1) + gx) * stride
                    pred_y = ((sig_dy * self.xy_scale) - 0.5 * (self.xy_scale - 1) + gy) * stride

                    # pred_wh = exp(dwdh) * anchors
                    pred_w = np.exp(dw) * anchor_w
                    pred_h = np.exp(dh) * anchor_h

                    predictions.append({
                        'x': pred_x,
                        'y': pred_y,
                        'w': pred_w,
                        'h': pred_h,
                        'objectness': objectness,
                        'class_probs': class_probs,
                    })

        return predictions

    def decode_outputs(self, outputs, org_h, org_w, resize_ratio, pad_w, pad_h, score_thresh: float | None = None):
        """
        Decode YOLO output layers and convert to pixel coordinates.

        Handles both YOLOv4 (3 outputs) and YOLOv8 (1 output) formats.

        Args:
            outputs: List of raw outputs from model
            org_h, org_w: Original frame dimensions
            resize_ratio: Scale factor from original to resized
            pad_w, pad_h: Letterbox padding offsets
            score_thresh: Detection score threshold

        Returns:
            (pixel_boxes, pixel_lms) where boxes are [x1, y1, x2, y2, score]
        """
        pixel_boxes = []
        pixel_lms = []  # Empty landmarks for person detector

        thresh = self.score_thresh if score_thresh is None else float(score_thresh)

        if self.variant == "v8":
            # YOLOv8 has single output with shape (batch, 5, num_predictions)
            pixel_boxes = self._decode_predictions_v8(outputs[0], org_h, org_w, resize_ratio, pad_w, pad_h)
            # Filter by score threshold
            pixel_boxes = [box for box in pixel_boxes if box[4] >= thresh]
        else:
            # YOLOv4 has 3 outputs (stride 32, 16, 8)
            for layer_idx, raw_output in enumerate(outputs):
                stride = self.strides[layer_idx]
                anchors_for_stride = self.anchors[layer_idx]

                predictions = self._decode_predictions(raw_output, stride, anchors_for_stride)

                for pred in predictions:
                    # Compute final score: objectness * max class probability
                    max_class_prob = float(np.max(pred['class_probs']))
                    score = pred['objectness'] * max_class_prob

                    if score < thresh:
                        continue

                    # Filter by person class (class 0)
                    if int(np.argmax(pred['class_probs'])) != self.target_class_id:
                        continue

                    # Convert xywh -> x1y1x2y2 in letterbox space
                    x_center = pred['x']
                    y_center = pred['y']
                    box_w = pred['w']
                    box_h = pred['h']

                    x1_letterbox = x_center - box_w / 2
                    y1_letterbox = y_center - box_h / 2
                    x2_letterbox = x_center + box_w / 2
                    y2_letterbox = y_center + box_h / 2

                    # Reverse letterbox padding
                    x1 = (x1_letterbox - pad_w) / resize_ratio
                    y1 = (y1_letterbox - pad_h) / resize_ratio
                    x2 = (x2_letterbox - pad_w) / resize_ratio
                    y2 = (y2_letterbox - pad_h) / resize_ratio

                    # Clip to original image bounds
                    x1 = max(0, min(x1, org_w - 1))
                    y1 = max(0, min(y1, org_h - 1))
                    x2 = max(0, min(x2, org_w - 1))
                    y2 = max(0, min(y2, org_h - 1))

                    # Skip invalid boxes
                    if x2 <= x1 or y2 <= y1:
                        continue

                    pixel_boxes.append([x1, y1, x2, y2, score])

        return pixel_boxes, pixel_lms

    def nms(self, pixel_boxes, pixel_lms, iou_threshold=None):
        """
        Apply Non-Maximum Suppression using OpenCV.

        Args:
            pixel_boxes: List of [x1, y1, x2, y2, score]
            pixel_lms: List of landmarks (unused for person detector)
            iou_threshold: IoU threshold for NMS

        Returns:
            (kept_boxes, kept_lms)
        """
        if not pixel_boxes:
            return [], []

        if iou_threshold is None:
            iou_threshold = self.nms_iou

        bboxes = []
        scores = []
        for (x1, y1, x2, y2, s) in pixel_boxes:
            bboxes.append([float(x1), float(y1), float(x2 - x1), float(y2 - y1)])  # xywh
            scores.append(float(s))

        idx = cv2.dnn.NMSBoxes(bboxes, scores, score_threshold=0.0, nms_threshold=iou_threshold)
        idx = idx.flatten().tolist() if len(idx) else []

        kept_boxes = [pixel_boxes[i] for i in idx]
        kept_lms = [pixel_lms[i] for i in idx] if pixel_lms else []

        return kept_boxes, kept_lms

    def __call__(self, frame: np.ndarray, threshold: float = 0.5):
        """
        CenterFace-compatible callable interface.

        Args:
            frame: Input frame (H, W, 3) in BGR format
            threshold: Score threshold for detections

        Returns:
            (dets, lms) where:
            - dets: (N, 5) array of [x1, y1, x2, y2, score]
            - lms: (N, 10) array of landmarks (empty for person detector)
        """
        outputs, org_h, org_w, resize_ratio, pad_w, pad_h = self.infer(frame)
        pixel_boxes, pixel_lms = self.decode_outputs(
            outputs, org_h, org_w, resize_ratio, pad_w, pad_h,
            score_thresh=threshold
        )
        kept_boxes, kept_lms = self.nms(pixel_boxes, pixel_lms)

        # Return in standard format
        dets = np.asarray(kept_boxes, dtype=np.float32) if kept_boxes else np.zeros((0, 5), np.float32)
        lms = np.asarray(kept_lms, dtype=np.float32) if kept_lms else np.zeros((0, 10), np.float32)

        return dets, lms
