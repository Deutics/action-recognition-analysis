import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
import cv2
from utils.logger import get_logger

from .backend import PoseBackend


logger = get_logger(__name__)


class TensorRTBackend(PoseBackend):

    def __init__(self, engine_path: str):
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, "rb") as f:
            runtime = trt.Runtime(self.logger)
            self.engine = runtime.deserialize_cuda_engine(f.read())

        self.context = self.engine.create_execution_context()

        self.input_idx = self.engine.get_binding_index("images")
        self.output_idx = 1

        self.input_shape = self.engine.get_binding_shape(self.input_idx)

        self.input_size = trt.volume(self.input_shape) * np.float32().nbytes
        self.output_size = trt.volume(self.engine.get_binding_shape(self.output_idx)) * np.float32().nbytes

        self.d_input = cuda.mem_alloc(self.input_size)
        self.d_output = cuda.mem_alloc(self.output_size)

        self.bindings = [int(self.d_input), int(self.d_output)]

    def preprocess(self, frame):
        frame = cv2.resize(frame, (640, 640))
        frame = frame[:, :, ::-1]
        frame = frame.transpose(2, 0, 1)
        frame = frame.astype(np.float32) / 255.0
        frame = np.expand_dims(frame, axis=0)
        return np.ascontiguousarray(frame)

    def infer(self, frame):

        input_tensor = self.preprocess(frame)

        cuda.memcpy_htod(self.d_input, input_tensor)

        self.context.execute_v2(self.bindings)

        output = np.empty(self.engine.get_binding_shape(self.output_idx), dtype=np.float32)
        cuda.memcpy_dtoh(output, self.d_output)

        # You will need to adapt postprocessing here
        return output, []