import cv2
import numpy as np
import pyrealsense2.pyrealsense2 as rs
<<<<<<< HEAD
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
=======
import argparse

# Define command-line arguments
parser = argparse.ArgumentParser(description='Blue blob detection using RealSense camera')
parser.add_argument('--mode', type=str, default='color', choices=['color', 'ir', 'all'],
                    help='Detection mode: color (blue detection), ir (bright spot detection), or all')
args = parser.parse_args()

print("RealSense blob detection starting...")
>>>>>>> parent of 899ead5... test

# Color thresholds for blue object detection (HSV)
lower_blue = np.array([100, 150, 50])
upper_blue = np.array([140, 255, 255])

# IR thresholds for bright spot detection
ir_threshold = 200  # Adjust based on your IR camera's brightness

# Main processing loop
try:
    # Initialize RealSense pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable color stream
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Enable infrared stream (if needed)
    if args.mode == 'ir' or args.mode == 'all':
        config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)

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
        # Wait for a coherent pair of frames
        try:
            frames = pipeline.wait_for_frames()

            # Align frames if needed
            aligned_frames = align.process(frames)

            # Get color frame
            color_frame = aligned_frames.get_color_frame()

            # Get IR frame if needed
            if args.mode == 'ir' or args.mode == 'all':
                ir_frame = aligned_frames.get_infrared_frame(1)  # Get the first IR frame

            if not color_frame:
                continue

            # Convert frames to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())

<<<<<<< HEAD
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
=======
            # Make a copy for output display
            output_frame = color_image.copy()

            # Initialize blob counters for this frame
            color_blob_count = 0
            ir_blob_count = 0
            
            # Process frames based on mode
            if args.mode == 'color' or args.mode == 'all':
                # Try to detect blue objects in color mode
                try:
                    # Convert BGR image to HSV
                    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

                    # Create a mask for the blue color
                    color_mask = cv2.inRange(hsv_image, lower_blue, upper_blue)

                    # Apply morphological operations to remove noise
                    color_mask = cv2.erode(color_mask, None, iterations=2)
                    color_mask = cv2.dilate(color_mask, None, iterations=2)

                    # Find contours in the mask
                    contours, _ = cv2.findContours(color_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Process all sufficiently large contours (multiple blobs)
                    if len(contours) > 0:
                        # Process each contour that is large enough
                        for contour in contours:
                            area = cv2.contourArea(contour)
                            if area < 100:  # Minimum area threshold
                                continue
                                
                            ((x, y), radius) = cv2.minEnclosingCircle(contour)
                            M = cv2.moments(contour)

                            # Calculate the center of the contour (avoid division by zero)
                            if M["m00"] > 0:
                                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                                # Draw a circle around the blob only if it's above a certain size
                                if radius > 10:  # Minimum radius threshold
                                    color_blob_count += 1
                                    cv2.circle(output_frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
                                    cv2.circle(output_frame, center, 5, (0, 0, 255), -1)
                                    # Add blob ID text
                                    label = f"C#{color_blob_count}" if args.mode == "all" else f"#{color_blob_count}"
                                    cv2.putText(output_frame, label, (int(x), int(y) - int(radius) - 10),
                                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                    print(f"Color blob #{color_blob_count} detected at: {center}, radius: {radius:.1f}, area: {area:.1f}")
                except Exception as e:
                    print(f"Error in color processing: {e}")

            if args.mode == 'ir' or args.mode == 'all':
                # Try to detect bright spots in IR mode
                try:
                    # Convert IR frame to numpy array
                    ir_image = np.asanyarray(ir_frame.get_data())

                    # Threshold to get bright spots
                    _, ir_mask = cv2.threshold(ir_image, ir_threshold, 255, cv2.THRESH_BINARY)

                    # Apply morphological operations to remove noise
                    ir_mask = cv2.erode(ir_mask, None, iterations=1)
                    ir_mask = cv2.dilate(ir_mask, None, iterations=2)

                    # Find contours in the mask
                    contours, _ = cv2.findContours(ir_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Process all sufficiently large contours
                    for contour in contours:
                        # Get area and ignore small contours
                        area = cv2.contourArea(contour)
                        if area < 50:  # Minimum area threshold, adjust as needed
                            continue

                        ((x, y), radius) = cv2.minEnclosingCircle(contour)
                        M = cv2.moments(contour)

                        # Calculate the center of the contour (avoid division by zero)
                        if M["m00"] > 0:
                            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

                            # Draw on output frame
                            ir_blob_count += 1
                            cv2.circle(output_frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                            cv2.circle(output_frame, center, 3, (0, 0, 255), -1)
                            # Add blob ID text
                            label = f"I#{ir_blob_count}" if args.mode == "all" else f"IR #{ir_blob_count}"
                            cv2.putText(output_frame, label, (int(x), int(y) - int(radius) - 10),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            print(f"IR blob #{ir_blob_count} detected at: {center}, radius: {radius:.1f}, area: {area:.1f}")
                except Exception as e:
                    print(f"Error in IR processing: {e}")

            # Add text overlay with mode info and blob counts
            cv2.putText(output_frame, f"Mode: {args.mode}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display the counts
            if args.mode == 'color' or args.mode == 'all':
                cv2.putText(output_frame, f"Color blobs: {color_blob_count}", (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            if args.mode == 'ir' or args.mode == 'all':
                cv2.putText(output_frame, f"IR blobs: {ir_blob_count}", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Blob Detection', output_frame)

            # Display the mask for debug purposes
            if args.mode == 'color':
                cv2.imshow('Color Mask', color_mask)
            elif args.mode == 'ir':
                cv2.imshow('IR Mask', ir_mask)

            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
>>>>>>> parent of 899ead5... test
                break

        except Exception as e:
            print(f"Error during frame processing: {e}")
            # Continue trying to get frames
            continue

except Exception as e:
    print(f"Error during execution: {e}")

finally:
    # Stop streaming
    print("Stopping RealSense pipeline...")
    if 'pipeline' in locals():
        pipeline.stop()
    cv2.destroyAllWindows()
    print("Pipeline stopped.")
