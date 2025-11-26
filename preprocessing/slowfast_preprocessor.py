# preprocessing/slowfast_preprocessor.py
import cv2
import torch
import numpy as np

MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD  = np.array([0.225, 0.225, 0.225], dtype=np.float32)

class SlowFastPreprocessor:

    def __init__(self, fast_T: int, img_size: int, alpha: int = 4):
        self.fast_T = fast_T
        self.img_size = img_size
        self.alpha = alpha
        self.slow_T = fast_T // alpha

    # process ONE frame → same API as FramePreprocessor
    def preprocess_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.img_size, self.img_size))
        norm = resized.astype(np.float32) / 255.0
        norm = (norm - MEAN) / STD
        return norm

    # convert buffer to SlowFast input
    def make_tensor(self, frames):
        frames = frames[-self.fast_T:]

        if len(frames) < self.fast_T:
            last = frames[-1]
            frames = frames + [last] * (self.fast_T - len(frames))

        idxs = np.linspace(0, self.fast_T - 1, self.slow_T).astype(int)
        slow_path = [frames[i] for i in idxs]
        fast_path = frames

        slow_tensor = self._to_tensor(slow_path)
        fast_tensor = self._to_tensor(fast_path)

        return [slow_tensor, fast_tensor]

    def _to_tensor(self, frames):
        arr = np.stack(frames, axis=0)     # (T,H,W,C)
        arr = arr.transpose(3,0,1,2)       # (C,T,H,W)
        return torch.from_numpy(arr).float().unsqueeze(0)
