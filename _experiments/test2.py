#!/usr/bin/env python3.6
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Handles CUDA context initialization and cleanup
import numpy as np
from PIL import Image
import time

# --- Constants ---
ENGINE_PATH = 'best-ext.engine'  # Path to your TensorRT engine file
# Path to your input image
IMAGE_PATH = 'eval_images/DSC05557_JPG.rf.d5d6e841cedb52118b792ca72e0525f0.jpg'
# Input dimensions expected by the engine (must match the fixed shape it was built with)
INPUT_H = 640
INPUT_W = 640
# Number of classes the model was trained on (e.g., 80 for COCO)
NUM_CLASSES = 80
# Expected output layout: [cx, cy, w, h, confidence, class_prob_0, ..., class_prob_N-1]
OUTPUT_ELEMENTS_PER_BOX = 5 + NUM_CLASSES

# --- Image Preprocessing ---


def preprocess_image(image_path, input_h, input_w):
    """
    Loads an image, resizes and pads it to the required input size,
    normalizes it, and formats it for TensorRT inference.

    Args:
        image_path (str): Path to the input image.
        input_h (int): Target input height.
        input_w (int): Target input width.

    Returns:
        np.ndarray: Preprocessed image as a NumPy array (NCHW format).
        tuple: Original image dimensions (width, height).
        tuple: Resize ratio (width_ratio, height_ratio).
    """
    try:
        img = Image.open(image_path)
        original_w, original_h = img.size
        print(f"Original image size: {original_w}x{original_h}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, (0, 0), (0, 0)

    # Calculate resize ratio and padding
    ratio = min(input_w / original_w, input_h / original_h)
    new_w = int(original_w * ratio)
    new_h = int(original_h * ratio)
    ratio_w, ratio_h = ratio, ratio  # Store ratio for post-processing scaling

    # Resize the image
    # Use LANCZOS for better quality
    img_resized = img.resize((new_w, new_h))

    # Create a new image with padding
    # Use a common padding color like gray (114) which YOLO often uses
    padded_img = Image.new("RGB", (input_w, input_h), (114, 114, 114))
    dw = (input_w - new_w) // 2
    dh = (input_h - new_h) // 2
    padded_img.paste(img_resized, (dw, dh))

    # Convert to numpy array, normalize, and change layout
    # Convert to float32 first to avoid issues with uint8 division
    image_np = np.array(padded_img, dtype=np.float32)
    image_np /= 255.0  # Normalize to [0.0, 1.0]

    # Transpose from HWC (Height, Width, Channels) to CHW (Channels, Height, Width)
    image_np = image_np.transpose((2, 0, 1))

    # Add batch dimension -> NCHW
    image_np = np.expand_dims(image_np, axis=0)

    # Ensure the array is contiguous in memory
    image_np = np.ascontiguousarray(image_np)

    print(f"Preprocessed image shape: {image_np.shape}")
    return image_np, (original_w, original_h), (ratio_w, ratio_h)

# --- TensorRT Inference Helper Class ---


class TrtInference:
    """Helper class to manage TensorRT engine loading and inference."""

    def __init__(self, engine_path):
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        self.engine = self._load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

        # Store binding names and shapes for convenience
        self.input_names = [self.engine.get_binding_name(i) for i in range(
            self.engine.num_bindings) if self.engine.binding_is_input(i)]
        self.output_names = [self.engine.get_binding_name(i) for i in range(
            self.engine.num_bindings) if not self.engine.binding_is_input(i)]
        print(f"Engine Input Names: {self.input_names}")
        print(f"Engine Output Names: {self.output_names}")

    def _load_engine(self, engine_path):
        """Loads a TensorRT engine from file."""
        print(f"Loading engine from: {engine_path}")
        try:
            with open(engine_path, "rb") as f:
                engine_data = f.read()
            engine = self.runtime.deserialize_cuda_engine(engine_data)
            print("Engine loaded successfully.")
            assert engine is not None
            return engine
        except Exception as e:
            print(f"Error loading engine: {e}")
            raise SystemExit("Failed to load TensorRT engine.")

    def _allocate_buffers(self):
        """Allocates necessary host and device buffers."""
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        print("Allocating buffers...")
        for binding in self.engine:
            binding_idx = self.engine.get_binding_index(binding)
            # Get shape and dtype
            # Note: For fixed shape engines, get_binding_shape gives the actual shape
            shape = tuple(self.engine.get_binding_shape(binding_idx))
            dtype = trt.nptype(self.engine.get_binding_dtype(binding_idx))
            size = trt.volume(shape) * np.dtype(dtype).itemsize

            print(
                f"  Binding: {binding}, Index: {binding_idx}, Shape: {shape}, Dtype: {dtype}, Size: {size} bytes")

            # Allocate device memory
            device_mem = cuda.mem_alloc(size)
            bindings.append(int(device_mem))  # Address of device buffer

            # Allocate host memory (page-locked for async efficiency)
            host_mem = cuda.pagelocked_empty(trt.volume(shape), dtype=dtype)

            if self.engine.binding_is_input(binding_idx):
                inputs.append(
                    {'host': host_mem, 'device': device_mem, 'shape': shape, 'dtype': dtype})
            else:
                outputs.append(
                    {'host': host_mem, 'device': device_mem, 'shape': shape, 'dtype': dtype})

        print("Buffer allocation complete.")
        return inputs, outputs, bindings, stream

    def infer(self, input_image_np):
        """
        Runs inference on a preprocessed image.

        Args:
            input_image_np (np.ndarray): Preprocessed image (NCHW format).

        Returns:
            list: List of NumPy arrays containing the raw outputs from the engine.
        """
        if input_image_np.shape != self.inputs[0]['shape']:
            print(
                f"Error: Input image shape {input_image_np.shape} does not match engine input shape {self.inputs[0]['shape']}")
            # If batch size is dynamic but H/W are fixed, you might need context.set_binding_shape here
            return None

        # Copy input data from host buffer to device buffer asynchronously
        np.copyto(self.inputs[0]['host'],
                  input_image_np.ravel())  # Flatten input
        cuda.memcpy_htod_async(
            self.inputs[0]['device'], self.inputs[0]['host'], self.stream)

        # Run inference asynchronously
        # Use execute_async_v2 for TensorRT 7.0+
        if hasattr(self.context, "execute_async_v2"):
            self.context.execute_async_v2(
                bindings=self.bindings, stream_handle=self.stream.handle)
        else:  # Fallback for potentially older versions (less common now)
            self.context.execute_async(
                batch_size=1, bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output data from device buffer to host buffer asynchronously
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)

        # Synchronize the stream to wait for completion
        self.stream.synchronize()

        # Return the host buffers reshaped
        return [out['host'].reshape(out['shape']) for out in self.outputs]

    def destroy(self):
        """Manually clean up resources if needed (often handled by pycuda.autoinit)."""
        print("Cleaning up resources...")
        # You might uncomment buffer freeing if not using autoinit or facing memory issues
        # for inp in self.inputs:
        #     inp['device'].free()
        # for outp in self.outputs:
        #     outp['device'].free()
        # if self.stream:
        #     self.stream.synchronize() # Ensure stream is finished before freeing
        # Note: context, engine, runtime are usually managed ok by Python's GC + autoinit


# --- Main Execution ---

if __name__ == "__main__":
    print("Starting YOLOv5 TensorRT Inference...")

    # --- 1. Preprocess Image ---
    start_preprocess = time.time()
    input_image_np, original_dims, resize_ratio = preprocess_image(
        IMAGE_PATH, INPUT_H, INPUT_W)
    end_preprocess = time.time()

    if input_image_np is None:
        print("Failed to preprocess image. Exiting.")
        exit()

    print(
        f"Preprocessing time: {end_preprocess - start_preprocess:.4f} seconds")

    # --- 2. Initialize Inference Engine ---
    start_init = time.time()
    trt_infer = TrtInference(ENGINE_PATH)
    end_init = time.time()
    print(f"Engine initialization time: {end_init - start_init:.4f} seconds")

    # --- 3. Run Inference ---
    print("Running inference...")
    start_infer = time.time()
    raw_outputs = trt_infer.infer(input_image_np)
    end_infer = time.time()

    if raw_outputs:
        print(f"Inference time: {end_infer - start_infer:.4f} seconds")
        print(f"Received {len(raw_outputs)} output tensor(s).")

        # --- 4. Inspect Raw Output ---
        # Assuming the first output tensor contains the detections
        # Shape is likely (batch_size, num_boxes, num_elements_per_box)
        # e.g., (1, 25200, 85) for 640x640 input, 80 classes
        output_data = raw_outputs[0]
        # Should be like (1, N, 85)
        print(f"Raw output shape: {output_data.shape}")
        num_potential_boxes = output_data.shape[1]
        print(
            f"Number of potential boxes (before filtering/NMS): {num_potential_boxes}")

        # Print a small slice to see the structure
        print("Example raw output (first few boxes):")
        # Print batch 0, first 5 boxes, all elements
        print(output_data[0, :5, :])

        # --- !! IMPORTANT: Post-processing Needed !! ---
        print("\n" + "="*30)
        print("IMPORTANT: Raw Output Explanation")
        print("="*30)
        print("The output above is RAW data from the model.")
        print("Each row likely represents a potential bounding box with format:")
        print(
            f"  [center_x, center_y, width, height, object_confidence, class_prob_0, ..., class_prob_{NUM_CLASSES-1}]")
        print("\nTo get usable detections, you MUST perform post-processing:")
        print("1. Filter boxes based on a confidence threshold (e.g., > 0.25).")
        print("2. Convert box format (cx, cy, w, h) to corner format (x1, y1, x2, y2).")
        print(
            "3. Scale coordinates back to the original image size using the resize ratio.")
        print("4. Apply Non-Maximum Suppression (NMS) to remove overlapping boxes for the same object.")
        print("This script does NOT perform these steps.")
        print("="*30 + "\n")

    else:
        print("Inference failed.")

    # --- 5. Cleanup (Optional) ---
    # trt_infer.destroy() # Usually handled by pycuda.autoinit

    print("Inference script finished.")
