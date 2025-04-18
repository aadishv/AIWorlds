import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
# import pyrealsense2 as rs
from inference_engine import TRTInference
import time

# Path to your TensorRT engine file
ENGINE_PATH = "model.engine"  # Update this path to your actual engine file location

# Model configuration parameters (must match those in inference_engine.py)
INPUT_SHAPE = (3, 640, 640)  # (Channels, Height, Width)
NUM_CLASSES = 1
CONF_THRESH = 0.5  # Increased from 0.25 to 0.5
IOU_THRESH = 0.45

# Initialize TensorRT inference engine
print("Loading TensorRT engine...")
model = TRTInference(ENGINE_PATH, INPUT_SHAPE,
                     NUM_CLASSES, CONF_THRESH, IOU_THRESH)

print("RealSense object detection starting...")

# Main processing loop
try:
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Enable depth stream (still needed for alignment)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Get device info
    device = profile.get_device()
    device_product_line = str(device.get_info(rs.camera_info.product_line))
    print(f"Connected to {device_product_line}")

    align_to = rs.stream.color
    align = rs.align(align_to)

    print("RealSense pipeline started. Press 'q' to quit.")

    while True:
        # Wait for frames
        try:
            frames = pipeline.wait_for_frames()

            # Align frames
            aligned_frames = align.process(frames)

            # Get color frame
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # Store original image for display
            display_image = color_image.copy()

            # No need to resize here as that's handled in the TRTInferNet.preprocess method

            # Record inference start time
            t_infer_start = time.time()

            # Inference using TensorRT engine
            try:
                detections = model(color_image)  # Use __call__ method directly
                t_infer_end = time.time()
                inference_success = True
            except Exception as e:
                print(f"Error in TensorRT inference: {e}")
                t_infer_end = time.time()
                inference_success = False
                detections = []

            # Annotation and Display Prep
            t_annotate_start = time.time()
            if inference_success and detections:
                # Create a copy of the original image for annotation
                annotated_image = display_image.copy()

                # Draw bounding boxes and add labels
                detection_count = len(detections)
                for det in detections:
                    # det format: [x1, y1, x2, y2, confidence, class_id]
                    x1, y1, x2, y2, conf, class_id = det

                    # Convert to integers for drawing
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    # Draw rectangle
                    cv2.rectangle(annotated_image, (x1, y1),
                                  (x2, y2), (0, 255, 0), 2)

                    # Draw label background
                    label = f"Class {int(class_id)}: {conf:.2f}"
                    text_size = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
                    cv2.rectangle(
                        annotated_image,
                        (x1, y1 - text_size[1] - 4),
                        (x1 + text_size[0], y1),
                        (0, 255, 0),
                        -1
                    )

                    # Add text
                    cv2.putText(
                        annotated_image,
                        label,
                        (x1, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 0),
                        1
                    )

                # Add detection count
                cv2.putText(
                    annotated_image,
                    f"Objects detected: {detection_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )

                # Add inference time
                cv2.putText(
                    annotated_image,
                    f"Inference time: {(t_infer_end - t_infer_start)*1000:.1f} ms",
                    (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1,
                )
            else:
                annotated_image = display_image.copy()  # Show original if inference failed
                # Add text indicating no detections
                cv2.putText(
                    annotated_image,
                    "No detections",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2,
                )
            t_annotate_end = time.time()

            # Display
            cv2.imshow("Object Detection", annotated_image)
            key = cv2.waitKey(1) & 0xFF

            # Display time information
            t_frame_end = time.time()
            frame_time = (t_frame_end - t_infer_start) * 1000
            t_display_end = time.time()

            # Optional: Uncomment to print timing metrics
            # print(f"Infer: {(t_infer_end - t_infer_start)*1000:.1f} ms, "
            #       f"Annotate: {(t_annotate_end - t_annotate_start)*1000:.1f} ms, "
            #       f"Display: {(t_display_end - t_frame_end)*1000:.1f} ms, "
            #       f"Total: {frame_time:.1f} ms")

            if key == ord("q"):
                break

        except Exception as e:
            print(f"Error during frame processing: {e}")
            continue

except Exception as e:
    print(f"Error during execution: {e}")

finally:
    # Stop streaming
    print("Stopping RealSense pipeline...")
    if "pipeline" in locals():
        pipeline.stop()
    cv2.destroyAllWindows()
    print("Pipeline stopped.")
