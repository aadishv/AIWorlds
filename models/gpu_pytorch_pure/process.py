import os
import time
import logging

from flask import Flask, jsonify
import numpy as np
import torch
import torchvision

# Import your utility functions
from utils.utils import non_max_suppression, xywh2xyxy, box_iou

app = Flask(__name__)
LOGGER = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

#
# ============ ONE‐TIME INITIALIZATION ============
#
# Model output filename (assumed to be in the same directory or provide absolute path)
INPUT_FILENAME = 'trt_output.npy'

# NMS & device settings
DEVICE = torch.device('cpu')  # or 'cuda:0' if you like
CONF_THRES = 0.7
IOU_THRES = 0.5
CLASSES = None
AGNOSTIC_NMS = False
MAX_DET = 1000
NM = 0  # number of masks (0 if no segmentation masks)

# Class names mapping
CLASS_NAMES = {
    0: "blue ring",
    1: "goal",
    2: "red ring",
    3: "robot"
}

LOGGER.info(
    f"Starting Flask server with torch {torch.__version__} / torchvision {torchvision.__version__}")
LOGGER.info(f"Using device: {DEVICE}")

#
# ============ ROUTES ============
#


@app.route('/ping', methods=['GET'])
def ping():
    """
    Loads the .npy file, runs NMS, and returns detections + per‐class counts as JSON.
    """
    # 1. Load the Triton/TensorRT output
    if not os.path.exists(INPUT_FILENAME):
        return jsonify({
            "error": f"File not found: {INPUT_FILENAME}. Run your inference step first."
        }), 404

    try:
        raw_output = np.load(INPUT_FILENAME)
        # Expect shape [1, N, 9]
        assert raw_output.ndim == 3 and raw_output.shape[0] == 1 and raw_output.shape[2] == 9
    except Exception as e:
        LOGGER.exception("Failed loading or validating .npy:")
        return jsonify({"error": str(e)}), 500

    # 2. Convert to torch.Tensor on the correct device
    try:
        output_tensor = torch.from_numpy(raw_output).to(DEVICE)
    except Exception as e:
        LOGGER.exception("Failed converting to torch.Tensor:")
        return jsonify({"error": str(e)}), 500

    # 3. Run non‐max suppression
    try:
        dets = non_max_suppression(
            output_tensor,
            CONF_THRES,
            IOU_THRES,
            CLASSES,
            AGNOSTIC_NMS,
            max_det=MAX_DET,
            nm=NM
        )
    except Exception as e:
        LOGGER.exception("Error during NMS:")
        return jsonify({"error": str(e)}), 500

    # 4. Process the first (and only) batch element
    if not dets or dets[0] is None or dets[0].numel() == 0:
        # No detections
        return jsonify({
            "detections": [],
            "counts": {name: 0 for name in CLASS_NAMES.values()}
        })

    # Move to CPU / numpy
    dets_np = dets[0].cpu().numpy()  # shape [M, 6] => x1,y1,x2,y2,conf,cls

    # 5. Build the per‐detection list
    detections_list = []
    for x1, y1, x2, y2, conf, cls_idx in dets_np:
        cls_idx = int(cls_idx)
        detections_list.append({
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "confidence": float(conf),
            "class_index": cls_idx,
            "class_name": CLASS_NAMES.get(cls_idx, f"Unknown({cls_idx})")
        })

    # 6. Count per‐class
    class_indices = dets_np[:, 5].astype(int)
    unique, counts = np.unique(class_indices, return_counts=True)
    counts_dict = {CLASS_NAMES[c]: int(cnt) for c, cnt in zip(unique, counts)}

    # Ensure we report zero for classes not detected
    for idx, name in CLASS_NAMES.items():
        counts_dict.setdefault(name, 0)

    # 7. Return JSON
    return jsonify({
        "detections": detections_list,
        "counts": counts_dict
    })


#
# ============ ENTRY POINT ============
#
if __name__ == '__main__':
    # The server will reload by default if you run in debug mode.
    # In production, use gunicorn or similar.
    app.run(host='0.0.0.0', port=5000, debug=False)
