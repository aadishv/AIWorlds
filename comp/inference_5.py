import colorsys
import json
import logging
import threading
import time
import urllib.request

import cv2
import numpy as np
from flask import Flask, jsonify
from utils.inference import InferenceEngine
from utils.schema import Detection, WORKER_30_URL

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

JSON_LOCK = threading.Lock()
RAW_JSON_PATH = "raw_5.json"

app = Flask(__name__)


def get_img():
    """
    Fetch the latest image from the worker endpoint.
    """
    resp = urllib.request.urlopen(WORKER_30_URL + '/image.jpg')
    arr = np.frombuffer(resp.read(), dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return img


def inference_loop():
    """
    Background thread: load TRT engine, then loop forever grabbing images,
    doing inference, and dumping raw.json.
    """
    engine = InferenceEngine("inference/best.engine")
    print("🤖 TensorRT engine loaded.")

    # build a palette (same as your original)
    classes = engine.nms_params["classes"]
    class_colors = {}
    for i, cls_name in enumerate(classes):
        h = float(i) / len(classes)
        r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
        # (B, G, R) for OpenCV
        class_colors[i] = (int(b * 255), int(g * 255), int(r * 255))

    try:
        while True:
            try:
                img = get_img()
            except Exception as e:
                print("⚠️ could not fetch image:", e)
                time.sleep(0.1)
                continue

            try:
                results: list[Detection] = engine.run(img) or []
                payload = {"stuff": results}

                # write out raw.json under lock
                with JSON_LOCK:
                    with open(RAW_JSON_PATH, "w") as f:
                        json.dump(payload, f)
            except Exception as e:
                print("⚠️ inference error:", e)
                # swallow and keep going
                time.sleep(0.01)
                continue
    finally:
        engine.close()
        print("🛑 Inference engine closed.")


@app.route("/detections", methods=["GET"])
def get_detection():
    """
    Return the most recent raw.json as JSON.
    """
    with JSON_LOCK:
        try:
            with open(RAW_JSON_PATH, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            data = {"stuff": []}
    # flask.jsonify will set the Content-Type and do a safe dump
    return jsonify(data)


if __name__ == "__main__":
    # Start inference background thread
    t = threading.Thread(target=inference_loop, daemon=True)
    t.start()
    app.run(host="0.0.0.0", port=4000, debug=False)
