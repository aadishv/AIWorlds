# main.py
import cv2
import numpy as np
import pyrealsense2 as rs
import tensor

print("RealSense object detection starting...")

ENGINE_PATH = '/home/aadish/AIWorlds/models/gpu_pytorch_pure/best.engine'

# Colors for up to 20 classes
# You can expand/modify as needed
CLASS_COLORS = [
    (0, 255,   0),
    (0,   0, 255),
    (255,   0,   0),
    (255, 255,   0),
    (255,   0, 255),
    (0, 255, 255),
    (128,   0, 128),
    (128, 128,   0),
    (0, 128, 128),
    (128, 128, 128),
]


def draw_detections(image, detections):
    """
    detections: list of dicts with keys:
      x1, y1, x2, y2, confidence, class_index, class_name
    """
    h, w = image.shape[:2]
    for det in detections:
        x1, y1 = int(det['x1']), int(det['y1'])
        x2, y2 = int(det['x2']), int(det['y2'])
        cls = det['class_index']
        name = det['class_name']
        conf = det['confidence']

        color = CLASS_COLORS[cls % len(CLASS_COLORS)]

        # Draw box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = "{}:{:.2f}".format(name, conf)
        (txt_w, txt_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        cv2.rectangle(
            image,
            (x1, y1 - txt_h - baseline),
            (x1 + txt_w, y1),
            color,
            cv2.FILLED
        )
        # Draw label text
        cv2.putText(
            image,
            label,
            (x1, y1 - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 0),
            1
        )

    return image


def main():
    with tensor.TRTInference(ENGINE_PATH) as runner:
        try:
            # 1) Start RealSense
            pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
            config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
            profile = pipeline.start(config)
            dev = profile.get_device()
            print("Connected to", dev.get_info(rs.camera_info.product_line))

            align = rs.align(rs.stream.color)
            print("Streaming started. Press 'q' to quit.")

            while True:
                # 2) Grab a frame
                frames = pipeline.wait_for_frames()
                aligned = align.process(frames)
                cframe = aligned.get_color_frame()
                if not cframe:
                    continue

                img = np.asanyarray(cframe.get_data())
                # Resize to network input size
                inp_w, inp_h = 640, 640
                img_resized = cv2.resize(img, (inp_w, inp_h))

                # 3) Inference
                try:
                    result = runner.run(img_resized)
                except Exception as e:
                    print("Inference error:", e)
                    result = None

                # 4) Draw boxes + labels
                out_img = img_resized.copy()
                if result and 'detections' in result:
                    dets = result['detections']  # list of dicts
                    out_img = draw_detections(out_img, dets)

                    # Overlay counts
                    counts = result.get('counts', {})
                    count_str = '  '.join(
                        ["{}:{}".format(k, v) for k, v in counts.items()]
                    )
                    text = "Total {}   {}".format(len(dets), count_str)
                    cv2.putText(
                        out_img, text, (10, inp_h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 2
                    )

                # 5) Display
                cv2.imshow("Object Detection", out_img)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except Exception as e:
            print("Runtime error:", e)

        finally:
            print("Stopping pipeline...")
            pipeline.stop()
            cv2.destroyAllWindows()
            print("Done.")


if __name__ == '__main__':
    main()
