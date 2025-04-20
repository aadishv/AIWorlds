# postprocess.py (or process.py)
import torch
import torchvision
import numpy as np
import time
import logging
# Make sure you copy the non_max_suppression function and its helpers
from utils.utils import non_max_suppression, xywh2xyxy, box_iou # Add any needed utils

LOGGER = logging.getLogger(__name__)

# --- Load Raw Data ---
input_filename = 'trt_output.npy'
try:
    output_data = np.load(input_filename)
    print(f"Loaded TRT output from {input_filename}, Shape: {output_data.shape}")
    assert len(output_data.shape) == 3 and output_data.shape[0] == 1 and output_data.shape[2] == 9, "Unexpected shape in loaded data"
except FileNotFoundError:
    print(f"Error: {input_filename} not found. Run the inference script (tensor.py) first.")
    exit()
except Exception as e:
    print(f"Error loading or verifying {input_filename}: {e}")
    exit()

# --- Perform NMS ---
device = torch.device('cpu')
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') # Optional GPU
print(f"Using device for NMS processing: {device}")

output_tensor = torch.from_numpy(output_data).to(device)
print(f"Converted output type: {type(output_tensor)}, Device: {output_tensor.device}")

conf_thres = 0.7
iou_thres = 0.5
classes = None
agnostic_nms = False
max_det = 1000

try:
    print(f"Running NMS with torch {torch.__version__} and torchvision {torchvision.__version__}")
    final_detections = non_max_suppression(output_tensor,
                                           conf_thres,
                                           iou_thres,
                                           classes,
                                           agnostic_nms,
                                           max_det=max_det,
                                           nm=0) # nm=0 masks
except Exception as e:
    print(f"Error during NMS processing: {e}")
    print("Ensure torch and torchvision are correctly installed and compatible in this Python environment.")
    exit()

# --- Process final_detections ---
if final_detections and final_detections[0] is not None:
    # Detections for the first (and only) image in the batch
    # Move to CPU and convert back to NumPy for easier handling
    detections_for_image0 = final_detections[0].cpu().numpy() # Shape: [N, 6] -> [x1, y1, x2, y2, confidence, class_index]

    print(f"Detected {len(detections_for_image0)} objects in total after NMS.")

    # **** START: Added code for counting per class ****
    if len(detections_for_image0) > 0:
        # Extract the class index column (the 6th column, index 5)
        # Convert indices to integers
        detected_class_indices = detections_for_image0[:, 5].astype(int)

        # Count occurrences of each unique class index
        unique_classes, counts = np.unique(detected_class_indices, return_counts=True)

        # --- IMPORTANT: Define your class names here ---
        # Create a dictionary mapping the class index (0, 1, 2, 3) to your actual class names
        # Make sure the indices match how your model was trained!
        class_names = {
            0: "blue ring",   # <-- REPLACE THIS
            1: "goal",   # <-- REPLACE THIS
            2: "red ring",   # <-- REPLACE THIS
            3: "robot"    # <-- REPLACE THIS
        }
        # --- End of class name definition ---

        print("\nCounts per class:")
        found_something = False
        # Create a dictionary to store counts, initializing known classes to 0
        class_counts = {idx: 0 for idx in class_names.keys()}
        # Update counts for detected classes
        for class_index, count in zip(unique_classes, counts):
             class_counts[class_index] = count

        # Print counts for all known classes
        for class_index, count in class_counts.items():
             class_name = class_names.get(class_index, f"Unknown Class {class_index}") # Get name or use index safely
             print(f"  - {class_name}: {count}")
             if count > 0:
                 found_something = True

        if not found_something:
            print("  (No objects found meeting thresholds for known classes)")

    else:
        # This case should technically not happen if len(detections_for_image0) > 0,
        # but good for completeness.
        print("\nCounts per class:")
        print("  (No objects detected)")

    # **** END: Added code for counting per class ****

    # Add your code here to scale boxes back to original image size, draw boxes, etc.
    # You can still iterate through 'detections_for_image0' for drawing:
    # for det in detections_for_image0:
    #    x1, y1, x2, y2, conf, cls_idx = det
    #    # ... scale coordinates ...
    #    # ... draw box ...

else:
    print("No objects detected after NMS.")
