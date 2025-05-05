import logging
import numpy as np
import torch
from torchvision.ops import nms

LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

INPUT_FILENAME = "trt_output.npy"
CONF_THRES = 0.3
IOU_THRES = 0.5
MAX_DET = 1000

CLASS_NAMES = {
    0: "blue ring",
    1: "goal",
    2: "red ring",
    3: "robot"
}

LOGGER.info("Starting post‐processing on CPU")


def xywh2xyxy(boxes):
    # boxes: Tensor[M,4] = [xc,yc,w,h]
    x_c, y_c, w, h = boxes.unbind(1)
    return torch.stack((x_c - w/2, y_c - h/2, x_c + w/2, y_c + h/2), dim=1)


def ping(raw):
    try:
        assert raw.ndim == 3 and raw.shape[0] == 1
    except Exception:
        LOGGER.exception("Failed loading .npy")
        return None

    # 2) to torch on CPU
    pred = torch.from_numpy(raw)

    # 3) NMS
    det = cpu_nms(pred)  # [K,6] tensor on CPU
    if det.numel() == 0:
        # no detections
        return None

    # 4) to NumPy for easy indexing
    det_np = det.numpy()  # [[x1,y1,x2,y2,score,cls], ...]

    # 5) build per‐detection list
    detections = []
    for x1, y1, x2, y2, conf, cls in det_np:
        cls = int(cls)
        detections.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(conf),
            "class_index": cls,
            "class_name": CLASS_NAMES.get(cls, f"Unknown({cls})")
        })

    # 6) per‐class counts
    idxs, cnts = np.unique(det_np[:, 5].astype(int), return_counts=True)
    counts = {CLASS_NAMES[i]: int(c) for i, c in zip(idxs, cnts)}
    # ensure zero for missing classes
    for _, name in CLASS_NAMES.items():
        counts.setdefault(name, 0)

    # 7) return JSON
    return detections
