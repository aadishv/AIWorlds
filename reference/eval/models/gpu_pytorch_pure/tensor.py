import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import cv2
from process import ping
import time


class TRTInference:
    """
    TensorRT inference wrapper with explicit resource management.

    Usage (option A):
        runner = TRTInference('best.engine')
        runner.run('my.jpg')
        runner.close()

    Usage (option B, context‐manager):
        with TRTInference('best.engine') as runner:
            runner.run('my.jpg')
    """
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

    def __init__(self,
                 engine_path: str,
                 input_shape=(1, 3, 640, 640),
                 output_shape=(1, 25200, 9),
                 device_id=0):
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.output_shape = output_shape

        # 1) Initialize CUDA driver & create context
        cuda.init()
        self.device = cuda.Device(device_id)
        # make_context() pushes the new context on the stack
        self.cuda_ctx = self.device.make_context()

        # 2) Load TRT engine
        self.engine = self._load_engine(engine_path)
        if self.engine is None:
            raise SystemExit(f"Failed to load engine '{engine_path}'")

        # 3) Allocate buffers + stream
        self.inputs, self.outputs, self.bindings, self.stream = \
            self._allocate_buffers(self.engine)

        # 4) Create TRT execution context
        self.trt_context = self.engine.create_execution_context()

    def _load_engine(self, path: str):
        with open(path, 'rb') as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            return runtime.deserialize_cuda_engine(f.read())

    def _allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append({'host': host_mem, 'device': device_mem})
            else:
                outputs.append({'host': host_mem, 'device': device_mem})

        return inputs, outputs, bindings, stream

    def _do_inference(self):
        # 1) H2D
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp['device'], inp['host'], self.stream)

        # 2) Run
        self.trt_context.execute_async_v2(
            self.bindings, stream_handle=self.stream.handle)

        # 3) D2H
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        # 4) Sync
        self.stream.synchronize()

        # Return host pointers
        return [out['host'] for out in self.outputs]

    def close(self):
        # 1) sync & free your PyCUDA streams/bindings
        self.stream.synchronize()
        self.stream = None
        for entry in (self.inputs or []) + (self.outputs or []):
            entry['device'].free()
        self.inputs = None
        self.outputs = None

        # 2) explicitly drop TRT objects while context is still current
        #    this will invoke their __del__ (and hence free any TRT internal
        #    allocations) under the proper context
        del self.trt_context
        del self.engine
        self.trt_context = None
        self.engine = None

        # 3) now it’s safe to pop the context
        self.cuda_ctx.pop()
        self.cuda_ctx = None

    def __enter__(self):
        # For use with `with TRTInference(...) as runner:`
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Clean up on exit from `with` block
        self.close()

    def run(self, img, output_filename: str = 'trt_output.npy'):
        # --- Preprocess Image ---
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))[None, ...]  # NHWC->NCHW
        img = np.ascontiguousarray(img)

        # Copy to host buffer
        np.copyto(self.inputs[0]['host'], img.ravel())

        # --- Inference ---
        trt_outs = self._do_inference()

        # --- Postprocess & reshape ---
        raw = trt_outs[0]
        try:
            out3d = raw.reshape(self.output_shape)
            # print(f"Reshaped raw output (size {raw.size}) into {out3d.shape}")
        except ValueError as e:
            print(
                f"ERROR: cannot reshape {raw.size} → {self.output_shape}: {e}")
            return

        # --- Save ---
        # np.save(output_filename, out3d)
        print("Pinging...")
        start = time.time()
        result = ping(out3d)
        if result is None:
            result = []
        print(len(result), "- took", time.time() - start, "seconds")
        return result


if __name__ == "__main__":
    import eval
    engine_file = '/home/aadish/AIWorlds/models/gpu_pytorch_pure/best.engine'

    # if len(sys.argv) < 2:
    #     print(f"Usage: python {sys.argv[0]} <image_path>")
    #     sys.exit(1)
    # img_file = sys.argv[1]
    with TRTInference(engine_file) as runner:
        # runner.run(img_file)
        eval.run_eval(lambda img: runner.run(cv2.imread(img)))
