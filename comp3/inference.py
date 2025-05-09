import time

import cv2
import numpy as np
import pycuda.driver as cuda
import tensorrt as trt
import torch
import torchvision as vision

from constants import ENGINE_PATH


class InferenceEngine:
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

    def __init__(
        self,
        engine_path: str,
        nms_params={
            "conf_thres": 0.40,
            "iou_thres": 0.5,
            "max_det": 50,
            "classes": ["blue", "bot", "goal", "red"],
            # "classes": ["blue", "goal", "red", "stake?", "robot"],
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
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
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
        if img is None:
            # Return a black image of the correct size if input is None
            return np.zeros(self.input_shape[1:], dtype=np.float32).transpose((2,0,1))
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))[None, ...]  # NHWC->NCHW
        img = np.ascontiguousarray(img)

        # Copy to host buffer
        np.copyto(self.inputs[0]["host"], img.ravel())
        return img # Return preprocessed image for batched inference

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
            d = {
                "x": float((x1 + x2) / 2),
                "y": float((y1 + y2) / 2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "class": self.nms_params["classes"][int(cls)],
                "depth": float(0),
                "confidence": float(conf),
            }
            detections.append(d)
        return detections

    # MARK: - main entry point
    def run(self, img):
        # every frame:
        # preprocessing -> input -> tensorrt -> output -> postprocessing (nms) -> json blob
        preprocessed_img = self._do_preprocessing(img)
        # For single image inference, we still need to copy to device
        np.copyto(self.inputs[0]["host"], preprocessed_img.ravel())

        trt_outs = self._do_inference()

        raw = trt_outs[0]
        try:
            out3d = raw.reshape(self.output_shape)
        except ValueError as e:
            print(
                f"ERROR: cannot reshape {raw.size} → {self.output_shape}: {e}")
            return []
        return self._do_nms(out3d) or []


class BatchedInferenceEngine:
    TRT_LOGGER = trt.Logger(trt.Logger.ERROR)

    def __init__(
        self,
        engine_path: str,
        nms_params={
            "conf_thres": 0.40,
            "iou_thres": 0.5,
            "max_det": 50,
            "classes": ["blue", "bot", "goal", "red"],
        },
        input_shape=(2, 3, 640, 640),
        output_shape=(2, 25200, 9),
        device_id=0,
    ):
        self.engine_path = engine_path
        self.input_shape = input_shape
        self.output_shape = output_shape
        self.nms_params = nms_params
        cuda.init()
        self.device = cuda.Device(device_id)
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
        if img is None:
            return np.zeros((3, self.input_shape[2], self.input_shape[3]), dtype=np.float32)
        img = cv2.resize(img, (self.input_shape[3], self.input_shape[2]))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = img.transpose((2, 0, 1))  # HWC->CHW
        return img

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
        self.stream.synchronize()
        self.stream = None
        for entry in (self.inputs or []) + (self.outputs or []):
            entry["device"].free()
        self.inputs = None
        self.outputs = None

        del self.trt_context
        del self.engine
        self.trt_context = None
        self.engine = None

        self.cuda_ctx.pop()
        self.cuda_ctx = None

    def _xywh2xyxy(self, boxes):
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
        x = pred.squeeze(0)
        if x.numel() == 0:
            return x.new_zeros((0, 6))

        obj_conf = x[:, 4]
        cls_conf_vals, cls_idx = x[:, 5:].max(dim=1)
        scores = obj_conf * cls_conf_vals

        mask = scores > conf_thres
        if not mask.any():
            return x.new_zeros((0, 6))

        x = x[mask]
        scores = scores[mask]
        cls_idx = cls_idx[mask]

        boxes = self._xywh2xyxy(x[:, :4])

        keep = vision.ops.nms(boxes, scores, iou_thres)
        if keep.numel() > max_det:
            keep = keep[:max_det]

        det = torch.cat(
            (
                boxes[keep],
                scores[keep].unsqueeze(1),
                cls_idx[keep].unsqueeze(1).float(),
            ),
            dim=1,
        )

        return det

    def _do_nms(self, out3d):
        det = self._cpu_nms(torch.from_numpy(out3d))
        if det.numel() == 0:
            return None
        det = det.numpy()
        detections = []
        for x1, y1, x2, y2, conf, cls in det:
            d = {
                "x": float((x1 + x2) / 2),
                "y": float((y1 + y2) / 2),
                "width": float(x2 - x1),
                "height": float(y2 - y1),
                "class": self.nms_params["classes"][int(cls)],
                "depth": float(0),
                "confidence": float(conf),
            }
            detections.append(d)
        return detections

    def run_batch(self, img1, img2):
        # Preprocess both images
        pre_img1 = self._do_preprocessing(img1)
        pre_img2 = self._do_preprocessing(img2)
        batch = np.stack([pre_img1, pre_img2], axis=0)
        # Copy to host buffer
        np.copyto(self.inputs[0]["host"], batch.ravel())
        trt_outs = self._do_inference()
        raw = trt_outs[0]
        try:
            out3d_batch = raw.reshape(self.output_shape)
        except ValueError as e:
            print(
                f"ERROR: cannot reshape {raw.size} → {self.output_shape}: {e}")
            return [], []
        detections1 = self._do_nms(out3d_batch[0:1]) or []
        detections2 = self._do_nms(out3d_batch[1:2]) or []
        return detections1, detections2


class InferenceWorker:
    def __init__(self, app):
        self.camera = app.camera
        self.raw_detections = [] # For RealSense camera
        self.raw_detections_2 = [] # For Jetson CSI camera
        # Make sure to use correct input/output shapes for batch=2, 640x640
        self.engine = BatchedInferenceEngine(
            ENGINE_PATH,
            input_shape=(2, 3, 640, 640),
            output_shape=(2, 25200, 9)
        )
        # self.camera.frames is (realsense_color, realsense_depth, jetson_csi_color)

    def worker(self):
        self.engine.cuda_ctx.push()
        try:
            while True:
                realsense_img = None
                jetson_csi_img = None

                with self.camera._frames_lock: # Assuming CameraWorker has _frames_lock
                    if len(self.camera.frames) > 0:
                        realsense_img = self.camera.frames[0]
                    if len(self.camera.frames) > 2:
                        jetson_csi_img = self.camera.frames[2]
                
                # If jetson_csi_img is None, pass a black image or handle appropriately
                if jetson_csi_img is None:
                    # Create a black image of expected size if jetson_csi_img is not available
                    # This helps prevent errors if the CSI camera isn't ready yet
                    # Adjust shape as per your CSI camera's output if different from RealSense
                    if realsense_img is not None:
                         jetson_csi_img = np.zeros_like(realsense_img)
                    else: # if both are None, we can't do much, so skip or send two black images
                         jetson_csi_img = np.zeros((self.engine.input_shape[2], self.engine.input_shape[3], 3), dtype=np.uint8)


                if realsense_img is None: # If realsense is also None, make it black
                    realsense_img = np.zeros((self.engine.input_shape[2], self.engine.input_shape[3], 3), dtype=np.uint8)


                detections1, detections2 = self.engine.run_batch(realsense_img, jetson_csi_img)
                self.raw_detections = detections1
                self.raw_detections_2 = detections2
                
                time.sleep(0.01) # Adjust sleep time as needed
        finally:
            # pop when you exit, so you don’t leak
            self.engine.cuda_ctx.pop()
            self.engine.close()
