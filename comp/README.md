To run, run `python3.6 worker_30.py` and `python3.6 inference_5.py` in different terminal tabs. Read from the brain or go to localhost:5000 for the dashboard.

**Files**
There is some fundamental utility stuff that is long enough to put in its own section:

- Under the inference folder there is a file which works with TensorRT runtime to run our YOLO model
- Under the serial folder there is a file with classes that handle serial communication with the V5 brain

There are only three other code files:

- worker\_30 is the big dog, the most important file. It runs at 30 fps (thus the name) and directly handles communications with the Realsense. It also exposes an API including a web dashboard for debugging and monitoring and an endpoint to get the most recent frame from the camera (used in multiple places). It turns out that up to three different processes may simultaneously access the camera, so worker\_30 must be able to handle multiple concurrent requests, so we run its Flask app with multiple threads. It *also* interfaces with the serial communication classes in the serial folder to post serialized data to the V5 brain.
- inference\_5.py isn't expected to run very quickly, and it can dip as low as 5 fps. This interfaces with code in the inference folder to fetch the latest frame from worker_30, run the model, run post-processing (NMS), and return the results. It also exposes an API returning raw bounding boxes.

**Communications**

A lot of different methods of communication were tried, but I always ended up using the same wonderful one: *webservers*. Most API points are documented above.

Files are only used in the file scope; that is, the worker thread and api threads of each file share data over files instead of HTTP.

**TODOs**

- Implement Oliver's depth stuff (TBD) (test sending over whole row)
- Implement depth detection (steal VEX's code)
- Get serial working on worker_30's serialization
