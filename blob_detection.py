import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
import os
from datetime import datetime
from inference import get_model
import supervision as sv
import time

# Initialize Roboflow model
print("Loading Roboflow model...")
model = get_model(model_id="high-stakes-wnyrk/1")

# Create Supervision annotators
bounding_box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

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

            # --- Add resizing before inference ---
            # Example: Resize to 416x416 (common size for YOLO models) or smaller
            infer_width = 100
            infer_height = 100
            # Maintain aspect ratio if desired, or just resize
            color_image = cv2.resize(color_image, (infer_width, infer_height))

            # Inference
            try:
                results = model.infer(color_image)[0]
                detections = sv.Detections.from_inference(results)
                labels = [p.class_name for p in results.predictions]
            except Exception as e:
                print(f"Error in model inference: {e}")
                results = None  # Handle case where inference fails

            # Annotation and Display Prep
            if results:  # Only annotate if inference succeeded
                annotated_image = bounding_box_annotator.annotate(
                    scene=color_image.copy(), detections=detections
                )
                annotated_image = label_annotator.annotate(
                    scene=annotated_image, detections=detections, labels=labels
                )
                detection_count = len(results.predictions)
                cv2.putText(
                    annotated_image,
                    f"Objects detected: {detection_count}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                )
            else:
                annotated_image = (
                    color_image.copy()
                )  # Show original if inference failed

            # Display
            cv2.imshow("Object Detection", annotated_image)
            key = cv2.waitKey(1) & 0xFF

            # Print timings (optional, can be noisy)
            # print(f"Wait: {(t_wait_end - t_wait_start)*1000:.1f} ms, Align: {(t_align_end - t_align_start)*1000:.1f} ms, Infer: {(t_infer_end - t_infer_start)*1000:.1f} ms, Annotate: {(t_annotate_end - t_annotate_start)*1000:.1f} ms, Display: {(t_display_end - t_display_start)*1000:.1f} ms, Total: {(t_end - t_start)*1000:.1f} ms")

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
