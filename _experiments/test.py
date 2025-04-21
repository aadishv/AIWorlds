import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit  # Important for initializing CUDA context

# Define necessary parameters
onnx_file_path = 'best-ext.onnx'
engine_file_path = 'best-ext.engine'
precision = trt.DataType.FLOAT  # Or trt.DataType.HALF for FP16
max_batch_size = 1  # Or your desired max batch size if using dynamic batch
workspace_size_gib = 1  # Adjust as needed (GiB)

# --- CORRECTED WORKSPACE SETTING ---
# Convert GiB to Bytes for the older API
# 1 << 30 is equivalent to 1024*1024*1024
workspace_size_bytes = workspace_size_gib * (1 << 30)

# Create a TensorRT logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)  # Or INFO, VERBOSE

# Initialize TensorRT builder, network, and parser
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, TRT_LOGGER)

# Create a builder config
config = builder.create_builder_config()

# --- SET MAX WORKSPACE SIZE USING OLDER API ---
config.max_workspace_size = workspace_size_bytes
# --- END CORRECTION ---

# Set precision
if precision == trt.DataType.HALF:
    if not builder.platform_has_fast_fp16:
        print("WARNING: FP16 not supported on this platform.")
    else:
        config.set_flag(trt.BuilderFlag.FP16)
elif precision == trt.DataType.INT8:
    # INT8 requires calibration - more complex setup needed
    # config.set_flag(trt.BuilderFlag.INT8)
    # Add INT8 calibrator here
    print("INT8 calibration not implemented in this example.")
    exit()

print(f'Loading ONNX file from path {onnx_file_path}...')
with open(onnx_file_path, 'rb') as model:
    print('Beginning ONNX file parsing')
    if not parser.parse(model.read()):
        print('ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        exit()
print('Completed parsing of ONNX file')

# Define optimization profile for dynamic shapes if needed
profile = builder.create_optimization_profile()
input_name = network.get_input(0).name
# Define min/opt/max shapes: (batch, channels, height, width)
profile.set_shape(input_name, (1, 3, 640, 640),
                  (max_batch_size, 3, 640, 640), (max_batch_size, 3, 640, 640))
config.add_optimization_profile(profile)

print(
    f'Building an engine from file {onnx_file_path}; this may take a while...')
# Use build_serialized_network if available, fallback to build_engine otherwise
# Check TensorRT version or use hasattr for compatibility if unsure
if hasattr(builder, "build_serialized_network"):
    print("Using build_serialized_network")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        print("ERROR: Failed to build the engine using build_serialized_network.")
        exit()
    engine_data = serialized_engine  # Keep variable name consistent for saving
else:
    # Older TensorRT might only have build_engine
    print("Using build_engine (might be deprecated)")
    engine = builder.build_engine(network, config)
    if engine is None:
        print("ERROR: Failed to build the engine using build_engine.")
        exit()
    # Serialize the engine manually if needed for saving
    serialized_engine = engine.serialize()
    if serialized_engine is None:
        print("ERROR: Failed to serialize the engine.")
        exit()
    engine_data = serialized_engine  # Keep variable name consistent for saving
    # Explicitly destroy the non-serialized engine object if needed
    # del engine


print("Completed creating Engine")

# Save the engine
with open(engine_file_path, "wb") as f:
    f.write(engine_data)  # Use the serialized data
print(f"Engine saved to {engine_file_path}")

# Clean up (optional, happens on exit)
# del parser
# del network
# del config
# del builder
# del serialized_engine # or engine if using build_engine
