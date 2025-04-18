import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Manages CUDA context
import numpy as np
import cv2
import time
import sys

# --- Configuration ---
# Path to your TensorRT engine file on the Jetson Nano
ENGINE_PATH = 'model.engine'

# Model input dimensions (Channels, Height, Width) - MUST match your ONNX export and training size (640)
INPUT_SHAPE = (3, 640, 640)

# Number of classes your model was trained on
# You might need to verify this based on your model training configuration.
# The YOLOv5 output tensor shape is typically (Batch, Number of anchors * Grid cells, 5 + num_classes)
# If your OUTPUT_SHAPE is (1, 25200, N), then num_classes = N - 5
NUM_CLASSES = 1  # <--- ADJUST THIS BASED ON YOUR DATASET

# Confidence and IoU thresholds for filtering detections
CONF_THRESH = 0.25
IOU_THRESH = 0.45

# --- TensorRT Initialization ---
# Use WARNING for less verbose output during inference
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def load_engine(engine_path):
    """Load the TensorRT engine from a file."""
    print(f"Loading engine from {engine_path}")
    try:
        with open(engine_path, 'rb') as f:
            runtime = trt.Runtime(TRT_LOGGER)
            engine = runtime.deserialize_cuda_engine(f.read())
        print("Engine loaded successfully.")
        return engine
    except Exception as e:
        print(f"Error loading engine: {e}")
        return None


def allocate_buffers(engine):
    """Allocates host and device buffers for inputs and outputs."""
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * \
            engine.get_binding_dtype(binding).itemsize
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_buffer = cuda.pagelocked_empty(size // dtype.itemsize, dtype)
        device_buffer = cuda.mem_alloc(size)

        bindings.append(int(device_buffer))

        if engine.binding_is_input(binding):
            inputs.append({'host': host_buffer, 'device': device_buffer,
                          'shape': engine.get_binding_shape(binding)})
        else:
            outputs.append({'host': host_buffer, 'device': device_buffer,
                           'shape': engine.get_binding_shape(binding)})

    print("Input and output buffers allocated.")
    return inputs, outputs, bindings, stream


def preprocess_image(image_np, input_shape):
    """
    Preprocesses a NumPy image array for TensorRT inference.
    Resizes, normalizes, and changes channel order.
    Assumes input_shape is (C, H, W).
    """
    # Resize image to input shape (excluding channel)
    input_h, input_w = input_shape[1:]
    # YOLOv5 uses letterboxing for aspect ratio matching, but for simplicity
    # here we'll just resize directly. If performance or slight accuracy
    # differences are critical, consider implementing letterboxing.
    resized_image = cv2.resize(image_np, (input_w, input_h))

    # Normalize pixel values to [0, 1]
    normalized_image = resized_image.astype(np.float32) / 255.0

    # Convert HWC to CHW
    chw_image = np.transpose(normalized_image, (2, 0, 1))

    # Add batch dimension
    input_tensor = np.expand_dims(chw_image, axis=0)

    # Verify shape and dtype
    # Expected shape: (1, C, H, W)
    expected_shape = (1, *input_shape)
    if input_tensor.shape != expected_shape:
        print(
            f"Warning: Preprocessed input shape mismatch. Expected {expected_shape}, got {input_tensor.shape}")
    # Expected dtype: float32 (unless you built the engine with INT8)
    if input_tensor.dtype != np.float32:
        print(
            f"Warning: Preprocessed input dtype mismatch. Expected float32, got {input_tensor.dtype}. Engine might expect a different type.")

    return input_tensor


def postprocess_output(output_data, original_image_shape, input_shape, num_classes, conf_thresh, iou_thresh):
    """
    Postprocesses the raw TensorRT output for YOLOv5 detections.
    Decodes bounding boxes, applies confidence threshold, and performs NMS.
    """
    # Reshape the flattened output assuming standard YOLOv5 structure
    # Output shape from engine is typically (1, total_detections, 5 + num_classes)
    # total_detections for 640x640 is 25200 (80*80*3 + 40*40*3 + 20*20*3)
    # output_data is a list of arrays, typically one array for the main output
    raw_output = output_data[0]  # Get the main output array

    # Reshape assuming the flattened output format
    # The last dimension should be 5 (box+conf) + num_classes
    output_dim = 5 + num_classes
    # Calculate total detections from flattened shape
    total_detections = raw_output.shape[0] // output_dim

    if total_detections * output_dim != raw_output.shape[0]:
        print(
            f"Error: Raw output size ({raw_output.shape[0]}) is not a multiple of output dimension ({output_dim}). Check NUM_CLASSES.")
        return []

    # Reshape to (total_detections, 5 + num_classes)
    detections = raw_output.reshape(-1, output_dim)

    # Apply Sigmoid to confidence and class scores
    # Sigmoid for confidence and class scores
    detections[:, 4:] = 1 / (1 + np.exp(-detections[:, 4:]))

    # Filter detections by confidence threshold
    conf_mask = detections[:, 4] > conf_thresh
    detections = detections[conf_mask]

    if detections.shape[0] == 0:
        return []  # No detections found above confidence threshold

    # Get box coordinates (cx, cy, w, h) and confidence
    boxes_conf = detections[:, :5]
    # Get class scores
    class_scores = detections[:, 5:]

    # Get class with highest score and apply class score threshold implicitly
    # Filter by max class score > confidence threshold as well
    class_ids = np.argmax(class_scores, axis=1)
    class_confidences = np.max(class_scores, axis=1)

    # Combine confidence with class confidence (usually multiply)
    final_confidences = boxes_conf[:, 4] * class_confidences

    # Apply combined confidence threshold
    final_conf_mask = final_confidences > conf_thresh
    boxes_conf = boxes_conf[final_conf_mask]
    class_ids = class_ids[final_conf_mask]
    final_confidences = final_confidences[final_conf_mask]

    if boxes_conf.shape[0] == 0:
        return []  # No detections found above combined threshold

    # Convert cx, cy, w, h to x1, y1, x2, y2 (relative to input_shape)
    # YOLOv5 box decoding (simplified - does not include grid/anchor logic explicitly here,
    # assuming the ONNX export handles this and the output is already relative to the input grid/anchors
    # or the postprocessing is part of the ONNX graph and the output is already (x1, y1, x2, y2, conf, class_scores))
    # A standard YOLOv5 ONNX export *usually* outputs the raw predictions which need decoding
    # based on anchors and grid structure *outside* of the ONNX graph for older TRT versions.
    # If your export script handled this, the output might already be in a processed format.
    # Assuming the common case where decoding is needed:

    # This part is tricky without the grid/anchor information from the model definition.
    # A standard YOLOv5 ONNX export from export.py (without plugins) gives raw outputs
    # from the three detection heads, potentially concatenated.
    # The decoding requires knowing the anchor boxes for each scale and the grid cell coordinates.

    # Given the provided trtexec log shows Range/Reshape/Transpose nodes before the error,
    # it's likely the ONNX graph from export.py *does* contain the decoding logic up to
    # a point, and the final output tensor `onnx::Sigmoid_...` is after sigmoid on
    # the combined predictions from all scales/anchors, possibly reshaped.
    # A common output shape after some onnx-time postprocessing is (1, num_total_predictions, 5+num_classes),
    # where num_total_predictions is the sum of all anchor box predictions across grid scales.
    # Let's assume the `detections` array is already in a format where the first 4 columns
    # can be converted to x1, y1, x2, y2 directly after applying scaling and zero-point shifts if any.
    # Standard YOLOv5 output is (cx, cy, w, h) *before* scaling.

    # Let's assume the columns are already scaled or need scaling based on the input size
    # and are (cx, cy, w, h) relative to the input_shape (640x640)
    # (cx, cy, w, h) relative to input size (e.g., 640)
    boxes = np.copy(boxes_conf[:, :4])

    # Convert center coordinates and width/height to corner coordinates (x1, y1, x2, y2)
    boxes[:, 0] = boxes_conf[:, 0] - boxes_conf[:, 2] / 2  # x1 = cx - w/2
    boxes[:, 1] = boxes_conf[:, 1] - boxes_conf[:, 3] / 2  # y1 = cy - h/2
    boxes[:, 2] = boxes_conf[:, 0] + boxes_conf[:, 2] / 2  # x2 = cx + w/2
    boxes[:, 3] = boxes_conf[:, 1] + boxes_conf[:, 3] / 2  # y2 = cy + h/2

    # Clip boxes to image bounds (0 to input_size)
    boxes[:, 0] = np.clip(boxes[:, 0], 0, input_shape[2]
                          )  # Clip x1 to [0, input_w]
    boxes[:, 1] = np.clip(boxes[:, 1], 0, input_shape[1]
                          )  # Clip y1 to [0, input_h]
    boxes[:, 2] = np.clip(boxes[:, 2], 0, input_shape[2]
                          )  # Clip x2 to [0, input_w]
    boxes[:, 3] = np.clip(boxes[:, 3], 0, input_shape[1]
                          )  # Clip y2 to [0, input_h]

    # Apply Non-Maximum Suppression (NMS)
    # Simple NumPy NMS implementation
    def non_max_suppression_np(boxes, scores, iou_threshold):
        """Pure Python/NumPy implementation of Non-Maximum Suppression."""
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        areas = (x2 - x1) * (y2 - y1)
        # Get indices of scores in descending order
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)

            # Calculate intersection over union (IoU)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # Keep indices where overlap is less than the threshold
            inds = np.where(ovr <= iou_threshold)[0]
            # +1 because ovr is calculated against order[1:]
            order = order[inds + 1]

        return keep

    # Apply NMS to the filtered boxes
    # We apply NMS per class for better results, but a simpler version is global NMS.
    # Let's do global NMS first for simplicity, then explain per-class if needed.
    # Per-class NMS is more accurate but requires grouping boxes by class ID.

    # Global NMS
    # keep_indices = non_max_suppression_np(boxes, final_confidences, iou_thresh)

    # Per-class NMS (Recommended for better accuracy)
    keep_indices = []
    unique_classes = np.unique(class_ids)
    for class_id in unique_classes:
        class_mask = class_ids == class_id
        class_boxes = boxes[class_mask]
        class_scores = final_confidences[class_mask]
        # Original indices for this class
        class_indices = np.where(class_mask)[0]

        if class_boxes.shape[0] > 0:
            class_keep = non_max_suppression_np(
                class_boxes, class_scores, iou_thresh)
            # Add original indices of kept boxes
            keep_indices.extend(class_indices[class_keep])

    # Final detected boxes after NMS
    final_boxes = boxes[keep_indices]
    final_confidences = final_confidences[keep_indices]
    final_class_ids = class_ids[keep_indices]

    # Scale bounding box coordinates back to original image dimensions
    original_h, original_w = original_image_shape
    input_h, input_w = input_shape[1:]
    scale_x = original_w / input_w
    scale_y = original_h / input_h

    # Apply scaling to the final boxes (x1, y1, x2, y2)
    final_boxes[:, 0] *= scale_x
    final_boxes[:, 1] *= scale_y
    final_boxes[:, 2] *= scale_x
    final_boxes[:, 3] *= scale_y

    # Format the final output
    # List of lists/tuples: [x1, y1, x2, y2, confidence, class_id]
    detected_objects = []
    for i in range(final_boxes.shape[0]):
        x1, y1, x2, y2 = final_boxes[i].tolist()
        confidence = float(final_confidences[i])
        class_id = int(final_class_ids[i])
        detected_objects.append([x1, y1, x2, y2, confidence, class_id])

    return detected_objects


class TRTInference:
    """
    Wrapper class for TensorRT inference.
    """

    def __init__(self, engine_path, input_shape, num_classes, conf_thresh=0.25, iou_thresh=0.45):
        self.engine = load_engine(engine_path)
        if not self.engine:
            raise RuntimeError("Failed to load TensorRT engine.")

        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = allocate_buffers(
            self.engine)

        # Store config
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Verify input binding shape
        if self.inputs[0]['shape'][1:] != self.input_shape:
            print(
                f"Warning: Engine input shape {self.inputs[0]['shape']} does not match specified INPUT_SHAPE {self.input_shape}.")

    def __del__(self):
        """
        Cleans up TensorRT resources.
        """
        del self.stream
        del self.outputs
        del self.inputs
        del self.bindings
        del self.context
        del self.engine
        print("TensorRT resources cleaned up.")

    def __call__(self, image_np):
        """
        Performs inference on a single image (NumPy array).
        Args:
            image_np (np.ndarray): Input image as a NumPy array (HWC format).
        Returns:
            list: A list of detected objects, where each object is
                  [x1, y1, x2, y2, confidence, class_id].
        """
        if image_np is None:
            print("Error: Input image is None.")
            return []

        original_image_shape = image_np.shape[:2]  # (height, width)

        # Preprocess the image
        input_tensor = preprocess_image(image_np, self.input_shape)

        # Copy input tensor to device input buffer
        # self.inputs[0]['host'] is already pagelocked
        np.copyto(self.inputs[0]['host'], input_tensor.ravel())
        cuda.memcpy_htod_async(
            self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Execute the engine
        self.context.execute_async_v2(
            bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output from device output buffers to host output buffers
        output_data = []
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
            output_data.append(out['host'])

        # Synchronize the stream to ensure data is copied to host
        self.stream.synchronize()

        # Postprocess the output data
        detected_objects = postprocess_output(
            output_data,
            original_image_shape,
            self.input_shape,
            self.num_classes,
            self.conf_thresh,
            self.iou_thresh
        )

        return detected_objects


# --- Example Usage ---
if __name__ == '__main__':
    # --- IMPORTANT ---
    # 1. Make sure you have the necessary libraries installed on your Jetson Nano.
    # 2. Replace 'path/to/your/model.engine' with the actual path to your engine file.
    # 3. Replace 'path/to/your/sample_image.jpg' with the path to a sample image on your Jetson.
    # 4. **VERY IMPORTANT:** Adjust the `NUM_CLASSES` variable above to match the number of classes your model was trained on.

    engine_file_path = ENGINE_PATH  # Uses the constant defined above
    sample_image_path = 'path/to/your/sample_image.jpg'  # <--- REPLACE THIS

    # Create an instance of the TensorRT inference wrapper
    try:
        # Pass the engine path, input shape, and number of classes to the constructor
        trt_infer_net = TRTInference(
            engine_file_path, INPUT_SHAPE, NUM_CLASSES, CONF_THRESH, IOU_THRESH)
        print("TensorRT inference wrapper initialized.")
    except Exception as e:
        print(f"Failed to initialize TensorRT inference: {e}")
        sys.exit(1)

    # Load a sample image
    try:
        sample_image_np = cv2.imread(sample_image_path)
        if sample_image_np is None:
            raise FileNotFoundError(f"Image not found at {sample_image_path}")
        print(f"Loaded sample image with shape: {sample_image_np.shape}")
    except Exception as e:
        print(f"Error loading sample image: {e}")
        sys.exit(1)

    # Perform inference
    print("Running inference on sample image...")
    start_time = time.time()
    bounding_boxes = trt_infer_net(sample_image_np)
    end_time = time.time()
    print(f"Inference took {end_time - start_time:.4f} seconds.")

    # Print the detected bounding boxes
    print("\nDetected bounding boxes (x1, y1, x2, y2, confidence, class_id):")
    if bounding_boxes:
        for box in bounding_boxes:
            print(box)
    else:
        print("No objects detected above the confidence threshold.")

    # The trt_infer_net object will be cleaned up automatically when the script ends
