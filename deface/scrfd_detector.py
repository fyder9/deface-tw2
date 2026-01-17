
import cv2
import numpy as np 

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
        return frame, new_w / w, new_h / h
    
    def preprocess(self, img_rgb: np.ndarray) -> np.ndarray:
        #TODO:Convert input from RGB to BGR based on SCRFD requirements
        img_bgr=cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).astype(np.float32)
        img_bgr = (img_bgr -127.5) / 128.0
        img_bgr = np.transpose(img_bgr, (2, 0, 1))  # HWC to CHW
        img_bgr = np.expand_dims(img_bgr, axis=0)  # Add batch dimension
        return img_bgr
    
    def infer(self, frame: np.ndarray): #Perform raw inference on input frame
        frame = self.ensure_rgb(frame)
        frame, scale_w, scale_h = self.resize_frame(frame)
        blob = self.preprocess(frame)
        outputs = self.sess.run(self.output_names, {self.input_name: blob})
        return outputs, scale_w, scale_h
