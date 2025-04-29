import os
import cv2
import json
import time
import numpy as np
import tensorrt as trt
import pycuda.autoinit     # initializes CUDA driver
import pycuda.driver as cuda


class InferenceEngine:
    def __init__(self,
                 engine_path,
                 conf_thresh=0.25,
                 iou_thresh=0.45,
                 input_size=(640, 640),
                 profile=True):
        """
        engine_path: path to your .engine file
        conf_thresh: object confidence threshold
        iou_thresh: NMS IoU threshold
        input_size: (W, H) that the engine expects
        profile: if True, prints timing breakdown each run
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.input_w, self.input_h = input_size
        self.profile = profile

        # TRT logger, runtime, and engine deserialization
        self.logger = trt.Logger(trt.Logger.INFO)
        with open(engine_path, "rb") as f, trt.Runtime(self.logger) as runtime:
            self.engine = runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()

        # Allocate host/device buffers for each binding
        self.bindings = []
        self.stream = cuda.Stream()
        for idx in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(idx)
            dtype = trt.nptype(self.engine.get_binding_dtype(idx))
            shape = self.context.get_binding_shape(idx)
            size = abs(int(np.prod(shape)))
            # page‐locked host array + device buffer
            host_mem = cuda.pagelocked_empty(size, dtype)
            dev_mem = cuda.mem_alloc(host_mem.nbytes)
            self.bindings.append((name, host_mem, dev_mem))

        # after you know your output size, e.g.
        # output has shape (1, N, 5 + num_classes)
        # so:
        self.num_classes = self.context.get_binding_shape(1)[-1] - 5

    def _preprocess(self, image_path):
        img = cv2.imread(image_path)
        orig_h, orig_w = img.shape[:2]
        img_resized = cv2.resize(img, (self.input_w, self.input_h))
        img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
        img_norm = img_rgb.astype(np.float32) / 255.0
        img_chw = np.transpose(img_norm, (2, 0, 1))
        return img_chw, (orig_w, orig_h)

    def _postprocess(self, output, orig_size):
        """
        output: np.ndarray of shape (N, 5 + num_classes)
        orig_size: (orig_w, orig_h)
        """
        orig_w, orig_h = orig_size
        bboxes = []     # in (x,y,w,h) pixel coords
        scores = []
        class_ids = []

        # 1) Decode & filter by conf_thresh
        for det in output:
            x, y, w, h, conf = det[:5]
            class_conf = det[5:]
            class_id = int(class_conf.argmax())
            score = float(conf * class_conf[class_id])
            if score < self.conf_thresh:
                continue

            # Convert center->corner & rescale to original image
            x0 = (x - w/2) * (orig_w / self.input_w)
            y0 = (y - h/2) * (orig_h / self.input_h)
            w0 = w * (orig_w / self.input_w)
            h0 = h * (orig_h / self.input_h)
            # round to ints for OpenCV NMS
            bboxes.append([int(x0), int(y0), int(w0), int(h0)])
            scores.append(score)
            class_ids.append(class_id)

        # 2) If nothing survived threshold, return empty
        if not bboxes:
            return []

        # 3) Perform NMS per class with OpenCV (C++ speed)
        keep_all = []
        # cv2.dnn.NMSBoxes takes boxes in x,y,w,h format
        for cls in set(class_ids):
            cls_idxs = [i for i, c in enumerate(class_ids) if c == cls]
            cls_boxes = [bboxes[i] for i in cls_idxs]
            cls_scores = [scores[i] for i in cls_idxs]
            # returns list of [[idx],[idx],...] on success
            keep = cv2.dnn.NMSBoxes(
                cls_boxes,
                cls_scores,
                self.conf_thresh,
                self.iou_thresh
            )
            if len(keep) > 0:
                # flatten [[i],[j],...] -> [i,j,...]
                flat = [k[0] if isinstance(k, (list, tuple, np.ndarray)) else int(k)
                        for k in keep]
                # map back to global indices
                keep_all += [cls_idxs[k] for k in flat]

        # 4) Build the JSON‐style result
        results = []
        for i in keep_all:
            x, y, w, h = bboxes[i]
            cx = x + w/2
            cy = y + h/2
            results.append({
                "x": float(cx),
                "y": float(cy),
                "width": float(w),
                "height": float(h),
                "class": int(class_ids[i]),
                "confidence": float(scores[i]),
            })
        return results

    def run(self, image_path):
        # Optional CPU timing
        t0 = time.perf_counter()
        img_chw, orig_size = self._preprocess(image_path)
        t1 = time.perf_counter()

        # Copy to device
        _, host_in, dev_in = self.bindings[0]
        host_in[:] = img_chw.ravel()
        cuda.memcpy_htod_async(dev_in, host_in, self.stream)
        t2 = time.perf_counter()

        # GPU timing via CUDA Events
        start_evt = cuda.Event()
        end_evt = cuda.Event()
        start_evt.record(self.stream)

        # Inference
        binding_addrs = [int(b[2]) for b in self.bindings]
        self.context.execute_async_v2(
            bindings=binding_addrs, stream_handle=self.stream.handle)

        end_evt.record(self.stream)
        # Copy output back
        _, host_out, dev_out = self.bindings[1]
        cuda.memcpy_dtoh_async(host_out, dev_out, self.stream)
        # Wait for everything
        self.stream.synchronize()
        t3 = time.perf_counter()

        gpu_time = start_evt.time_till(end_evt)  # ms

        # Reshape output tensor
        out_shape = self.context.get_binding_shape(
            1)  # e.g. (1, N, 6+num_classes)
        output = np.array(host_out).reshape(out_shape)[0]
        t4 = time.perf_counter()

        # Postprocess
        results = self._postprocess(output, orig_size)
        t5 = time.perf_counter()

        if self.profile:
            print(f"[PROFILE] preprocess: {(t1-t0)*1e3:7.2f} ms")
            print(f"[PROFILE] h2d copy:   {(t2-t1)*1e3:7.2f} ms")
            print(f"[PROFILE] inference:  {gpu_time:7.2f} ms")
            print(f"[PROFILE] d2h copy+:  {(t3-t2)*1e3:7.2f} ms  (incl. sync)")
            print(f"[PROFILE] postproc:   {(t5-t4)*1e3:7.2f} ms")
            print(f"[PROFILE] total run:  {(t5-t0)*1e3:7.2f} ms\n")

        return results


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--engine", required=True, help="Path to .engine file")
    p.add_argument("--img", required=True, help="Path to input image")
    p.add_argument("--profile", action="store_true",
                   help="Print timing breakdown")
    args = p.parse_args()

    ie = InferenceEngine(args.engine, profile=args.profile)
    dets = ie.run(args.img)
    print(json.dumps(dets, indent=2))
