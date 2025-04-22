import json
import sys
import time

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision as vision
from schema import Detection


class InferenceEngine:
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
    # MARK: - Lifecycle-related fucntions

    def __init__(
        self,
        engine_path: str,
        nms_params={
            "conf_thres": 0.7,
            "iou_thres": 0.5,
            "max_det": 50,
            "classes": ["blue", "goal", "red", "bot"],
        },
        input_shape=(1, 3, 640, 640),
        output_shape=(1, 25200, 9),
        device_id=0,
    ):
        # data
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nms_params = nms_params
        # set up CUDA device
        cuda.init()
        self.device = cuda.Device(device_id)
        # make_context() pushes the new context on the stack
        self.cuda_ctx = self.device.make_context()

        with open(self.engine_path, "rb") as f, trt.Runtime(self.TRT_LOGGER) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise SystemExit(f"Failed to load engine '{engine_path}'")
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers(
            self.engine
        )
        self.trt_context = self.engine.create_execution_context()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # MARK: - Inference utilities
    def _allocate_buffers(self, engine):
        inputs, outputs, bindings = [], [], []
        stream = cuda.Stream()
        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            bindings.append(int(device_mem))

            if engine.binding_is_input(binding):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})

        return inputs, outputs, bindings, stream

    def _do_preprocessing(self, img):
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))[None, ...]  # NHWC->NCHW
        img = np.ascontiguousarray(img)

        # Copy to host buffer
        np.copyto(self.inputs[0]["host"], img.ravel())

    def _do_inference(self):
        for inp in self.inputs:
            cuda.memcpy_htod_async(inp["device"], inp["host"], self.stream)
        self.trt_context.execute_async_v2(
            self.bindings, stream_handle=self.stream.handle
        )
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out["host"], out["device"], self.stream)
        self.stream.synchronize()
        return [out["host"] for out in self.outputs]

    def close(self):
        # 1) sync & free your PyCUDA streams/bindings
        self.stream.synchronize()
        self.stream = None
        for entry in (self.inputs or []) + (self.outputs or []):
            entry["device"].free()
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

    # MARK: - NMS utilies
    def _xywh2xyxy(self, boxes):
        # boxes: Tensor[M,4] = [xc,yc,w,h]
        x_c, y_c, w, h = boxes.unbind(1)
        return torch.stack((x_c - w / 2, y_c - h / 2, x_c + w / 2, y_c + h / 2), dim=1)

    @torch.no_grad()
    def _cpu_nms(
        self,
        pred: torch.Tensor,
        conf_thres: float = None,
        iou_thres: float = None,
        max_det: int = None,
    ) -> torch.Tensor:
        if conf_thres is None:
            conf_thres = self.nms_params["conf_thres"]
        if iou_thres is None:
            iou_thres = self.nms_params["iou_thres"]
        if max_det is None:
            max_det = self.nms_params["max_det"]
        """
        pred: Tensor[1, N, 5+nc] = [xc, yc, w, h, obj_conf, cls_conf_0, ..., cls_conf_nc-1]
        returns: Tensor[K,6] = [x1,y1,x2,y2,score,cls_idx]
        """
        x = pred.squeeze(0)  # [N,5+nc]
        if x.numel() == 0:
            return x.new_zeros((0, 6))

        # 1) objectness × best class confidence
        obj_conf = x[:, 4]  # [N]
        cls_conf_vals, cls_idx = x[:, 5:].max(dim=1)  # both [N]
        scores = obj_conf * cls_conf_vals  # [N]

        # 2) threshold
        mask = scores > conf_thres
        if not mask.any():
            return x.new_zeros((0, 6))

        x = x[mask]
        scores = scores[mask]
        cls_idx = cls_idx[mask]

        # 3) to corner format
        boxes = self._xywh2xyxy(x[:, :4])  # [M,4]

        keep = vision.ops.nms(boxes, scores, iou_thres)
        if keep.numel() > max_det:
            keep = keep[:max_det]

        # 5) gather detections
        det = torch.cat(
            (
                boxes[keep],
                scores[keep].unsqueeze(1),
                cls_idx[keep].unsqueeze(1).float(),
            ),
            dim=1,
        )  # [K,6]

        return det

    def _do_nms(self, out3d):
        det = self._cpu_nms(torch.from_numpy(out3d))
        if det.numel() == 0:
            return None
        det = det.numpy()  # [[x1,y1,x2,y2,score,cls], ...]

        detections = []
        for x1, y1, x2, y2, conf, cls in det:
            cls = int(cls)
            d = Detection(
                x=(x1 + x2) / 2,
                y=(y1 + y2) / 2,
                width=x2 - x1,
                height=y2 - y1,
                cls=cls,
                depth=0,
            )
            detections.append(d.serialize())
        with open("detections.json", "w") as f:
            json.dump(detections, f)
        return detections

    # MARK: - main entry point
    def run(self, img):
        self._do_preprocessing(img)
        trt_outs = self._do_inference()

        raw = trt_outs[0]
        try:
            out3d = raw.reshape(self.output_shape)
        except ValueError as e:
            print(f"ERROR: cannot reshape {raw.size} → {self.output_shape}: {e}")
            return

        result = self._do_nms(out3d)
        return [] if result is None else result


# if __name__ == "__main__":
#     img_path = "/home/aadish/AIWorlds/eval/eval_images/" + sys.argv[1]
#     with InferenceEngine("best.engine") as runner:
#         start = time.time()
#         runner.run(cv2.imread(img_path))
