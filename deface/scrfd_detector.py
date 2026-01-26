import cv2
import numpy as np 
import math


class SCRFDdetector: #Low-level SCRFD ONNX runtime wrapper
    def __init__(self, model_path: str, device: str = "cpu", cap_long_side: int = 1920, override_execution_provider: str = None):
        import onnxruntime
        #from .scrfd import SCRFD
        self.cap_long_side = cap_long_side  # Maximum long side length for input frames
        providers = onnxruntime.get_available_providers()   # Get list of available providers & choose one
         # If no override, use all available providers
        if override_execution_provider is None:
            ort_providers = providers
        else:
            if override_execution_provider in providers:
                ort_providers = [override_execution_provider]
            else:
                raise ValueError(f"Requested execution provider '{override_execution_provider}' is not available. Available providers: {providers}")
        self.sess = onnxruntime.InferenceSession(model_path, providers=ort_providers)
        self.input_name = self.sess.get_inputs()[0].name
        self.output_names = [output.name for output in self.sess.get_outputs()]
        self.nms_iou = 0.4
        self.score_thresh = 0.2

    def ensure_rgb(self,img: np.ndarray) -> np.ndarray:
            
            if img.ndim == 2:  # 1-channel grayscale -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:  # 4-channel RGBA -> RGB
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            return img
    
    def resize_frame(self, frame: np.ndarray) -> tuple[np.ndarray, float, float]:
        #Resize input frame to have its long side equal to cap_long_side while maintaining aspect ratio
        h, w = frame.shape[:2]
        longside = max(h, w)
        if longside > self.cap_long_side:
            scale = self.cap_long_side / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        else:
            return frame, 1.0, 1.0
        #returning resized or original frame
        return frame, new_w , new_h, scale
    
    def preprocess(self, img_rgb: np.ndarray) -> np.ndarray:
        #TODO:Convert input from RGB to BGR based on SCRFD requirements
        img_bgr=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        img_bgr = (img_bgr -127.5) / 128.0
        img_bgr = np.transpose(img_bgr, (2, 0, 1))  # HWC to CHW
        img_bgr = np.expand_dims(img_bgr, axis=0)  # Add batch dimension
        return img_bgr
    
    def infer(self, frame: np.ndarray): #Perform raw inference on input frame
        frame = self.ensure_rgb(frame)
        frame, new_w, new_h, scale = self.resize_frame(frame)
        blob = self.preprocess(frame)
        print("input shapes:", self.sess.get_inputs()[0].shape)
        outputs = self.sess.run(self.output_names, {self.input_name: blob})
        return outputs, new_w, new_h, scale

    def decode_outputs(self, outputs, new_w, new_h, scale):
        pixel_boxes = []
        pixel_lms = []

        for scale_idx in range(3): # take scores, bboxes, kps for each scale
            scores = outputs[scale_idx]     
            bboxes = outputs[scale_idx+3]    
            kps = outputs[scale_idx+6]
            stride = 8 * (2 ** scale_idx)
            fm_h = int(np.ceil(new_h / stride)) #feature map height * stride 8,16,32
            fm_w = int(np.ceil(new_w / stride)) #feature map width

            for i in range(scores.shape[0]): # for each anchor point
                score = float(scores[i][0])
                if score > self.score_thresh: #filter by score threshold
                    px = (i % fm_w + 0.5) * stride #x coordinate of anchor point
                    py = (i // fm_w + 0.5) * stride #y coordinate of anchor point
                    dx1 = bboxes[i][0] 
                    dy1 = bboxes[i][1]
                    dx2 = bboxes[i][2]
                    dy2 = bboxes[i][3]
                    x1 = px - dx1 * stride
                    y1 = py - dy1 * stride
                    x2 = px + dx2 * stride
                    y2 = py + dy2 * stride
                    pixel_boxes.append([x1, y1, x2, y2, score])

                    lm = kps[i] #landmarks
                    lm_out = []
                    for j in range(5):
                        x = (px + float(lm[2*j])     * stride) / scale
                        ly = (py + float(lm[2*j + 1]) * stride) / scale
                        lm_out.extend([x, ly])
                    pixel_lms.append(lm_out)

        return pixel_boxes, pixel_lms

    def nms(self, pixel_boxes, pixel_lms, iou_threshold=0.4): #Apply NMS using OpenCV, already filtered boxes by decode_outputs
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
        kept_lms   = [pixel_lms[i] for i in idx] if pixel_lms is not None else None
        return kept_boxes, kept_lms

    def __call__(self, frame: np.ndarray, threshold: float = 0.5):
        """
        CenterFace-compatible callable wrapper.

        SCRFD ONNX models typically output feature maps (scores, bbox, kps heads),
        not decoded detections. Until proper decode+NMS is implemented, this
        wrapper only runs inference and returns empty detections while printing
        output tensor shapes once for debugging.
        """
        # Run inference & print output shapes
        outputs, new_w, new_h, scale = self.infer(frame)
        self.decode_outputs(outputs, new_w, new_h, scale)
        # Debug output shapes once
        if not hasattr(self, "_printed_shapes"):
            print("SCRFD raw outputs:")
            for i, o in enumerate(outputs):
                if isinstance(o, np.ndarray):
                    print(f"  [{i}] shape={o.shape} dtype={o.dtype}")
                else:
                    print(f"  [{i}] type={type(o)}")
            self._printed_shapes = True
            

        # Return empty detections until decode+NMS is implemented
        dets = np.zeros((0, 5), dtype=np.float32)
        lms = np.zeros((0, 10), dtype=np.float32)
        return dets, lms
        
