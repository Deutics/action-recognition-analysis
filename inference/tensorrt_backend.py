# inference/tensorrt_backend.py
import os
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

    def infer(self, input_array: np.ndarray) -> dict:
        """
        input_array should match engine input shape and dtype.
        Returns dict of output_name -> numpy array (reshaped).
        """
        in_name = self.input_names[0]
        in_shape = tuple(self.context.get_tensor_shape(in_name))
        in_dtype = self._np_dtype(self.engine.get_tensor_dtype(in_name))

        # Make contiguous + correct dtype
        x = np.ascontiguousarray(input_array, dtype=in_dtype)

        if x.size != self._volume(in_shape):
            raise ValueError(f"Input size mismatch. Expected {in_shape} ({self._volume(in_shape)} elems) got {x.shape} ({x.size} elems)")

        # Copy H->D
        np.copyto(self.host[in_name], x.ravel())
        cuda.memcpy_htod_async(self.device[in_name], self.host[in_name], self.stream)

        # Execute
        ok = self.context.execute_async_v3(self.stream.handle)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")

        # Copy D->H outputs
        outputs = {}
        for out_name in self.output_names:
            out_shape = tuple(self.context.get_tensor_shape(out_name))
            out_dtype = self._np_dtype(self.engine.get_tensor_dtype(out_name))

            cuda.memcpy_dtoh_async(self.host[out_name], self.device[out_name], self.stream)

            outputs[out_name] = (np.array(self.host[out_name], dtype=out_dtype)
                                 .reshape(out_shape)
                                 .copy())

        self.stream.synchronize()
        return outputs