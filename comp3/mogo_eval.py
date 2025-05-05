import os
import time
import datetime
import cv2
import numpy as np

# Assuming this script is run from the AIWorlds/comp3 directory
# Adjust paths if necessary
try:
    from inference import InferenceEngine
    from constants import ENGINE_PATH, MOGO_TEST_FILES
except ImportError:
    print("Error: Could not import necessary modules.")
    print("Please ensure 'inference.py' and 'constants.py' are in the same directory or accessible.")
    print("Make sure you are running this script from the 'AIWorlds/comp3' directory.")
    exit(1)

# --- Configuration ---
OUTPUT_DIR = '../data/ours_annot_517'
# Ensure the output directory exists relative to the script's assumed location
script_dir = os.path.dirname(os.path.abspath(__file__))
output_dir_abs = os.path.abspath(os.path.join(script_dir, OUTPUT_DIR))
os.makedirs(output_dir_abs, exist_ok=True)

# Colors for drawing (BGR)
CLASS_COLORS = {
    "blue": (255, 0, 0),
    "goal": (0, 255, 255),  # Yellow
    "red": (0, 0, 255),
    "bot": (0, 255, 0),
    "default": (255, 255, 255)  # White for unknown
}
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.5
FONT_THICKNESS = 1
BOX_THICKNESS = 2

# --- Main Evaluation Logic ---


def draw_detections(image, detections):
    """Draws bounding boxes and labels on the image."""
    img_h, img_w = image.shape[:2]
    for det in detections:
        # Extract info
        x_center = det['x']
        y_center = det['y']
        width = det['width']
        height = det['height']
        conf = det['confidence']
        cls_name = det['class']

        # Convert YOLO format (center, width, height) scaled to 640x640
        # back to image coordinates (top-left, bottom-right)
        # The inference engine scales input to 640x640, so detections are relative to that
        # We need to scale back to the original image dimensions
        scale_x = img_w / 640.0
        scale_y = img_h / 640.0

        x1 = int((x_center - width / 2) * scale_x)
        y1 = int((y_center - height / 2) * scale_y)
        x2 = int((x_center + width / 2) * scale_x)
        y2 = int((y_center + height / 2) * scale_y)

        # Clamp coordinates to image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(img_w - 1, x2)
        y2 = min(img_h - 1, y2)

        # Get color and label
        color = CLASS_COLORS.get(cls_name, CLASS_COLORS["default"])
        label = f"{cls_name}: {conf:.2f}"

        # Draw rectangle
        cv2.rectangle(image, (x1, y1), (x2, y2), color, BOX_THICKNESS)

        # Draw label background
        (label_width, label_height), baseline = cv2.getTextSize(
            label, FONT, FONT_SCALE, FONT_THICKNESS)
        # Ensure label doesn't go off screen top
        label_y1 = max(y1, label_height + 5)
        cv2.rectangle(image, (x1, label_y1 - label_height - baseline),
                      (x1 + label_width, label_y1), color, cv2.FILLED)

        # Draw label text
        cv2.putText(image, label, (x1, label_y1 - baseline//2), FONT,
                    FONT_SCALE, (0, 0, 0), FONT_THICKNESS, cv2.LINE_AA)

    return image


def run_mogo_eval():
    """Runs inference on MOGO_TEST_FILES, saves annotated images, and prints stats."""
    print(f"Using engine: {ENGINE_PATH}")
    print(f"Processing {len(MOGO_TEST_FILES)} test images...")
    print(f"Saving annotated images to: {output_dir_abs}")

    engine = None
    inference_times = []
    total_start_time = time.time()

    try:
        engine = InferenceEngine(ENGINE_PATH)

        for i, img_path_rel in enumerate(MOGO_TEST_FILES):
            img_path_abs = os.path.abspath(
                os.path.join(script_dir, img_path_rel))
            print(
                f"Processing image {i+1}/{len(MOGO_TEST_FILES)}: {img_path_rel}")

            if not os.path.exists(img_path_abs):
                print(
                    f"  WARNING: Image not found at {img_path_abs}. Skipping.")
                continue

            # Load image
            image = cv2.imread(img_path_abs)
            if image is None:
                print(
                    f"  WARNING: Failed to load image {img_path_abs}. Skipping.")
                continue

            # Run inference and time it
            inf_start_time = time.time()
            # engine.run expects a cv2 image (numpy array)
            detections = engine.run(image)
            inf_end_time = time.time()
            inference_times.append(inf_end_time - inf_start_time)

            # Annotate image
            annotated_image = draw_detections(
                image.copy(), detections)  # Draw on a copy

            # Save annotated image
            base_filename = os.path.basename(img_path_rel)
            output_filename = os.path.join(
                output_dir_abs, f"{os.path.splitext(base_filename)[0]}_annot.jpg")
            cv2.imwrite(output_filename, annotated_image)
            # print(f"  Saved annotated image to: {output_filename}")

    except Exception as e:
        print(f"\nAn error occurred: {e}")
    finally:
        if engine:
            print("Cleaning up inference engine...")
            engine.close()

    total_end_time = time.time()
    total_time_elapsed = total_end_time - total_start_time
    num_processed = len(inference_times)

    # --- Print Statistics ---
    print("\n--- Evaluation Summary ---")
    print(f"Evaluation completed on: {datetime.datetime.now()}")

    if num_processed > 0:
        total_inference_time = sum(inference_times)
        avg_inference_time = total_inference_time / num_processed
        fps = num_processed / total_inference_time

        print(
            f"Successfully processed: {num_processed}/{len(MOGO_TEST_FILES)} images")
        print(f"Total script execution time: {total_time_elapsed:.2f} seconds")
        print(
            f"Total inference time (model execution only): {total_inference_time:.2f} seconds")
        print(
            f"Average inference time per image: {avg_inference_time:.3f} seconds ({avg_inference_time*1000:.1f} ms)")
        print(f"Inference Images Per Second (FPS): {fps:.2f}")
    else:
        print("No images were successfully processed.")
        print(f"Total script execution time: {total_time_elapsed:.2f} seconds")

    print("--------------------------")


if __name__ == "__main__":
    run_mogo_eval()
