import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Important for initializing CUDA context
import numpy as np
import cv2  # For image loading/preprocessing

# Logger for TensorRT
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

engine_file_path = 'best.engine'
# Batch, Channel, Height, Width - MUST match engine
input_shape = (1, 3, 640, 640)
# Example for YOLOv5s (Batch, num_predictions, box_params+classes) - MUST match engine
output_shape = (1, 25200, 9)


def load_engine(engine_file_path):
    with open(engine_file_path, 'rb') as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(
            binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp['device'], inp['host'], stream)
     for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out['host'], out['device'], stream)
     for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out['host'] for out in outputs]


# --- Main Execution ---
engine = load_engine(engine_file_path)
if not engine:
    raise SystemExit("Failed to load engine")

inputs, outputs, bindings, stream = allocate_buffers(engine)
context = engine.create_execution_context()

# --- Preprocess Image ---
# Load image, resize to input_shape (e.g., 640x640), normalize (0-1), transpose (HWC->CHW)
# Example (adapt letterboxing/normalization as needed for your model)
img_path = '/home/aadish/AIWorlds/eval_images/video_mp4-0031_jpg.rf.d890e93de1247a578e3387ea9fe4f0fd.jpg'
img_orig = cv2.imread(img_path)
img_resized = cv2.resize(
    img_orig, (input_shape[3], input_shape[2]))  # W, H for resize
img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_normalized = img_rgb.astype(np.float32) / 255.0
img_transposed = img_normalized.transpose((2, 0, 1))  # HWC to CHW
img_expanded = np.expand_dims(img_transposed, axis=0)  # Add batch dimension
input_image = np.ascontiguousarray(img_expanded)  # Make contiguous

# Copy preprocessed image data to the allocated host buffer
np.copyto(inputs[0]['host'], input_image.ravel())

# --- Run Inference ---
trt_outputs = do_inference(context, bindings=bindings,
                           inputs=inputs, outputs=outputs, stream=stream)

# --- Postprocess Output ---
# --- Reshape and SAVE the output ---
# Define the correct output shape (Batch=1, Boxes=25200, Outputs/Box=9)
output_shape = (1, 25200, 9)

# Get the raw output buffer (might be flat)
raw_output_buffer = trt_outputs[0]

# --> ADD THIS RESHAPE STEP <--
try:
    # Reshape the flat buffer into the expected 3D shape
    output_data_reshaped = raw_output_buffer.reshape(output_shape)
    print(
        f"Reshaped raw output (size {raw_output_buffer.size}) into shape {output_data_reshaped.shape}")
except ValueError as e:
    print(
        f"ERROR: Failed to reshape raw output buffer (size {raw_output_buffer.size}) to target shape {output_shape}.")
    print(f"Error message: {e}")
    print("Ensure the output_shape variable matches the engine's actual output.")
    exit()  # Stop if reshape fails

# Save the RESHAPED 3D array
output_filename = 'trt_output.npy'
np.save(output_filename, output_data_reshaped)  # <-- Save the reshaped version
print(
    f"Saved TensorRT output (shape {output_data_reshaped.shape}) to {output_filename}")

"""
# --- Postprocess Output ---
# trt_outputs[0] will contain the raw output tensor.
# Reshape it according to your model's output shape (e.g., [batch, num_boxes, 5+num_classes])
# Apply NMS, scale bounding boxes, etc.
# (This part requires understanding YOLOv5's output format and implementing NMS)
output_data = trt_outputs[0].reshape(output_shape)
print("Raw TRT output shape:", output_data.shape)
# --- Postprocess Output ---
import torch # Make sure torch is imported
import time # Needed by the nms function

# Assuming 'output_data' is your NumPy array with shape (1, 25200, 9)
print(f"Raw TRT output type: {type(output_data)}, Shape: {output_data.shape}")

# 1. Choose the device for NMS computations
#    Using CPU is generally safe and avoids potential GPU memory issues during NMS
device = torch.device('cpu')
#    Alternatively, to use GPU if available:
# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Using device for NMS processing: {device}")

# 2. Convert the NumPy array to a PyTorch tensor
output_tensor = torch.from_numpy(output_data).to(device)
print(f"Converted output type: {type(output_tensor)}, Device: {output_tensor.device}")

# 3. Now you can call the NMS function with the tensor
conf_thres = 0.4  # Example confidence threshold
iou_thres = 0.5   # Example IoU threshold
classes = None     # Filter by specific classes, e.g. [0, 1]
agnostic_nms = False
max_det = 1000

# Ensure the helper functions like xywh2xyxy are also defined/imported
# You might need to define LOGGER if the function uses it for warnings
import logging
LOGGER = logging.getLogger(__name__) # Simple logger definition

final_detections = non_max_suppression(output_tensor,
                                       conf_thres,
                                       iou_thres,
                                       classes,
                                       agnostic_nms,
                                       max_det=max_det,
                                       nm=0) # nm=0 because you have no masks

# --- Process final_detections as shown previously ---
if final_detections and final_detections[0] is not None:
    detections_for_image0 = final_detections[0].cpu().numpy() # Move to CPU and convert back to NumPy if needed for drawing etc.
    print(f"Detected {len(detections_for_image0)} objects after NMS.")
    # Process detections_for_image0 (remember to scale boxes)
else:
    print("No objects detected after NMS.")
"""
