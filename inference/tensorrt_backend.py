# inference/tensorrt_backend.py
import os
import cv2
import numpy as np
import tensorrt as trt

# You need a CUDA mem allocator. PyCUDA is the most common on Jetson.
import pycuda.driver as cuda
import pycuda.autoinit  # noqa: F401

from utils.logger import get_logger
logger = get_logger(__name__)


class TensorRTBackend:
    """
    TensorRT 10+ backend (IO Tensor API, no get_binding_index).
    Works with engines built using explicit batch.
    """

    def __init__(self, engine_path: str):
        assert os.path.exists(engine_path), f"Engine not found: {engine_path}"
        self.engine_path = engine_path

        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.trt_logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()
        self.engine = self.runtime.deserialize_cuda_engine(engine_bytes)
        if self.engine is None:
            raise RuntimeError("Failed to deserialize TensorRT engine.")

        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context.")

        # Discover I/O tensor names (TRT10 way)
        self.input_names = []
        self.output_names = []

        # TRT10 exposes engine.num_io_tensors and engine.get_tensor_name(i)
        n_io = self.engine.num_io_tensors
        for i in range(n_io):
            name = self.engine.get_tensor_name(i)
            mode = self.engine.get_tensor_mode(name)  # INPUT / OUTPUT
            if mode == trt.TensorIOMode.INPUT:
                self.input_names.append(name)
            elif mode == trt.TensorIOMode.OUTPUT:
                self.output_names.append(name)

        if not self.input_names:
            raise RuntimeError("No INPUT tensors found in engine.")
        if not self.output_names:
            raise RuntimeError("No OUTPUT tensors found in engine.")

        logger.info(f"TensorRT engine loaded: {engine_path}")
        logger.info(f"INPUT tensors: {self.input_names}")
        logger.info(f"OUTPUT tensors: {self.output_names}")

        # Buffers
        self.bindings = {}  # name -> device ptr (int)
        self.host = {}      # name -> host np array
        self.device = {}    # name -> cuda mem

        self.stream = cuda.Stream()

        # Most YOLO engines have 1 input. We’ll support multiple, but your code likely uses 1.
        self._allocate_io()

    def _np_dtype(self, trt_dtype):
        return trt.nptype(trt_dtype)

    def _volume(self, shape):
        v = 1
        for d in shape:
            v *= int(d)
        return int(v)

    def _input_hw(self):
        in_name = self.input_names[0]
        in_shape = tuple(self.context.get_tensor_shape(in_name))
        if len(in_shape) != 4:
            raise RuntimeError(f"Unexpected TensorRT input shape: {in_shape}")
        _, _, input_h, input_w = in_shape
        return int(input_h), int(input_w)

    def _letterbox(self, image: np.ndarray, new_shape: tuple):
        """
        Resize with unchanged aspect ratio and symmetric padding.
        Returns image and transform metadata for reverse mapping.
        """
        src_h, src_w = image.shape[:2]
        dst_h, dst_w = new_shape

        scale = min(dst_w / max(src_w, 1), dst_h / max(src_h, 1))
        resized_w = max(1, int(round(src_w * scale)))
        resized_h = max(1, int(round(src_h * scale)))

        resized = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)

        pad_w = dst_w - resized_w
        pad_h = dst_h - resized_h
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        letterboxed = cv2.copyMakeBorder(
            resized,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            cv2.BORDER_CONSTANT,
            value=(114, 114, 114),
        )

        meta = {
            "src_w": src_w,
            "src_h": src_h,
            "scale": scale,
            "pad_left": pad_left,
            "pad_top": pad_top,
        }
        return letterboxed, meta

    def _map_keypoints_to_source(self, kpts: np.ndarray, meta: dict) -> np.ndarray:
        """
        Convert keypoints from model input space back to original frame space.
        """
        mapped = kpts.astype(np.float32).copy()
        scale = max(float(meta["scale"]), 1e-6)
        pad_left = float(meta["pad_left"])
        pad_top = float(meta["pad_top"])
        src_w = float(meta["src_w"])
        src_h = float(meta["src_h"])

        mapped[:, 0] = (mapped[:, 0] - pad_left) / scale
        mapped[:, 1] = (mapped[:, 1] - pad_top) / scale
        mapped[:, 0] = np.clip(mapped[:, 0], 0, max(src_w - 1.0, 0.0))
        mapped[:, 1] = np.clip(mapped[:, 1], 0, max(src_h - 1.0, 0.0))
        return mapped

    def _allocate_io(self):
        """
        Allocate host/device buffers for all IO tensors.
        NOTE: If shapes are dynamic, you must set input shape before allocating.
        """
        # If your engine has dynamic shapes, you MUST set them before allocation.
        # We'll try to detect dynamic dims (-1) and refuse with a clear error.
        for name in self.input_names + self.output_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._np_dtype(self.engine.get_tensor_dtype(name))

            if any(d < 0 for d in shape):
                raise RuntimeError(
                    f"Dynamic shape detected for tensor '{name}' shape={shape}. "
                    f"Set input shape first via set_input_shape(...) then allocate."
                )

            nbytes = self._volume(shape) * np.dtype(dtype).itemsize
            host_mem = cuda.pagelocked_empty(self._volume(shape), dtype=dtype)
            dev_mem = cuda.mem_alloc(nbytes)

            self.host[name] = host_mem
            self.device[name] = dev_mem
            self.bindings[name] = int(dev_mem)

        # Tell context where IO buffers live (TRT10 way)
        for name, ptr in self.bindings.items():
            self.context.set_tensor_address(name, ptr)

    def set_input_shape(self, input_name: str, shape: tuple):
        """
        Use this ONLY if your engine has dynamic shapes.
        After setting shapes, you should re-allocate buffers.
        """
        if input_name not in self.input_names:
            raise ValueError(f"{input_name} is not an input tensor. Inputs: {self.input_names}")

        ok = self.context.set_input_shape(input_name, shape)
        if not ok:
            raise RuntimeError(f"Failed to set input shape for {input_name} -> {shape}")

        # Reallocate all buffers now that shapes are known
        self._allocate_io()

    def infer(self, input_array: np.ndarray):
        """
        Expects BGR image (H,W,3) as numpy array.
        Returns:
            keypoints_data: np.ndarray (N, 17, 3)
            track_ids: None (TensorRT pose export has no tracker)
        """

        # -----------------------------
        # 0) Validate frame
        # -----------------------------
        if input_array is None:
            logger.warning("TensorRTBackend.infer() received None frame")
            return np.empty((0, 17, 3), dtype=np.float32), None

        if not isinstance(input_array, np.ndarray):
            logger.warning(f"TensorRTBackend.infer() received non-numpy frame: {type(input_array)}")
            return np.empty((0, 17, 3), dtype=np.float32), None

        if input_array.ndim != 3 or input_array.shape[2] != 3:
            logger.warning(f"TensorRTBackend.infer() invalid frame shape: {input_array.shape}")
            return np.empty((0, 17, 3), dtype=np.float32), None

        h, w, c = input_array.shape
        if h <= 0 or w <= 0:
            logger.warning(f"TensorRTBackend.infer() empty frame: {input_array.shape}")
            return np.empty((0, 17, 3), dtype=np.float32), None

        # GI appsink sometimes returns readonly / non-contiguous memory
        # Make it safe for OpenCV ops
        img = np.ascontiguousarray(input_array)
        if not img.flags.writeable:
            img = img.copy()

        # -----------------------------
        # 1) Preprocess (YOLO TRT)
        # -----------------------------
        input_h, input_w = self._input_hw()
        img, transform_meta = self._letterbox(img, (input_h, input_w))

        # BGR -> RGB
        img = img[:, :, ::-1]

        # Normalize
        img = img.astype(np.float32) / 255.0

        # HWC -> CHW
        img = np.transpose(img, (2, 0, 1))

        # Add batch
        img = np.expand_dims(img, axis=0)

        # -----------------------------
        # 2) Prepare input buffers
        # -----------------------------
        in_name = self.input_names[0]
        in_shape = tuple(self.context.get_tensor_shape(in_name))
        in_dtype = self._np_dtype(self.engine.get_tensor_dtype(in_name))

        x = np.ascontiguousarray(img, dtype=in_dtype)

        if x.size != self._volume(in_shape):
            raise ValueError(
                f"Input size mismatch. Expected {in_shape} "
                f"({self._volume(in_shape)} elems) got {x.shape} ({x.size} elems)"
            )

        np.copyto(self.host[in_name], x.ravel())
        cuda.memcpy_htod_async(self.device[in_name], self.host[in_name], self.stream)

        # -----------------------------
        # 3) Execute
        # -----------------------------
        ok = self.context.execute_async_v3(self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        # -----------------------------
        # 4) Copy outputs
        # -----------------------------
        outputs = {}
        for out_name in self.output_names:
            out_shape = tuple(self.context.get_tensor_shape(out_name))
            out_dtype = self._np_dtype(self.engine.get_tensor_dtype(out_name))

            cuda.memcpy_dtoh_async(self.host[out_name], self.device[out_name], self.stream)

            outputs[out_name] = (
                np.array(self.host[out_name], dtype=out_dtype)
                .reshape(out_shape)
                .copy()
            )

        self.stream.synchronize()

        # -----------------------------
        # 5) Postprocess (simple)
        # -----------------------------
        output = outputs[self.output_names[0]]

        # Handle unexpected shapes safely
        if output is None or output.size == 0:
            return np.empty((0, 17, 3), dtype=np.float32), None

        # typically (1, N, 56) or (1, N, 57) depending on export layout
        if output.ndim == 3:
            dets = output[0]
        elif output.ndim == 2:
            dets = output
        else:
            logger.warning(f"Unexpected TRT output shape: {output.shape}")
            return np.empty((0, 17, 3), dtype=np.float32), None

        persons = []
        conf_threshold = 0.30
        kpt_values = 17 * 3

        for det in dets:
            # Support common YOLO export layouts:
            # [x,y,w,h,conf,cls,kpts...] or [x,y,w,h,conf,kpts...]
            if det.shape[0] < (5 + kpt_values):
                continue

            conf = float(det[4])
            if conf < conf_threshold:
                continue

            if det.shape[0] >= (6 + kpt_values):
                kpt_start = 6
            else:
                kpt_start = 5

            kpt_end = kpt_start + kpt_values
            if det.shape[0] < kpt_end:
                continue

            kpts = det[kpt_start:kpt_end].reshape(17, 3)
            persons.append(self._map_keypoints_to_source(kpts, transform_meta))

        if not persons:
            return np.empty((0, 17, 3), dtype=np.float32), None

        return np.stack(persons, axis=0), None
