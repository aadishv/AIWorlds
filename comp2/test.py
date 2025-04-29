import onnxruntime
import numpy as np

# try:
# Define model path and input shape based on export parameters
onnx_model_path = "yolov5n-best.onnx"
batch_size = 1
input_height = 640
input_width = 640
input_channels = 3

# Create dummy input data matching the expected shape and type
dummy_input = np.random.randn(batch_size, input_channels, input_height, input_width).astype(np.float32)

# Prefer CUDAExecutionProvider if available and compatible
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
# If CUDAExecutionProvider causes issues (e.g., version mismatch), fall back to CPU
# providers = ['CPUExecutionProvider']

session = onnxruntime.InferenceSession(onnx_model_path, providers=providers)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name # Assumes single output

# Run inference
results = session.run([output_name], {input_name: dummy_input})

print(f"ONNX model loaded successfully.")
print(f"Input Name: {input_name}, Output Name: {output_name}")
print(f"ONNX output shape: {results[0].shape}")
# Expected shape: (1, 25200, 85) or similar

# except Exception as e:
#     print(f"Error during ONNX Runtime check: {e}")
