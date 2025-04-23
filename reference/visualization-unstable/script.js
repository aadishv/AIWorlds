/**
 * Captures a single frame from a MediaStreamTrack (video) using ImageCapture.
 * @param {MediaStreamTrack} track - The video track to capture from.
 * @returns {Promise<ImageBitmap>} A Promise that resolves with an ImageBitmap of the captured frame.
 * @throws {Error} If ImageCapture is not supported or capture fails.
 */
async function captureFrameWithImageCapture(track) {
  if (typeof ImageCapture === "undefined") {
    throw new Error("ImageCapture API is not supported in this browser.");
  }
  if (track.kind !== "video") {
    throw new Error("The provided track is not a video track.");
  }
  if (track.readyState !== "live") {
    throw new Error("Track is not live.");
  }

  try {
    const imageCapture = new ImageCapture(track);
    const imageBitmap = await imageCapture.grabFrame();
    return imageBitmap;
  } catch (error) {
    // Common errors: DOMException: The track is not capable of providing stills
    // or track is ended.
    throw new Error(`Failed to grab frame using ImageCapture: ${error}`);
  }
}

/**
 * Continuously captures frames from a MediaStreamTrack and updates a canvas
 * @param {MediaStreamTrack} videoTrack - The video track to capture from
 */
async function exampleUsageImageCapture(videoTrack) {
  console.log("Starting continuous frame capture...");

  // Log detailed track information
  console.log("CAMERA METADATA - Track Information:");
  console.log("Track ID:", videoTrack.id);
  console.log("Track Label:", videoTrack.label);
  console.log("Track Kind:", videoTrack.kind);
  console.log("Track Enabled:", videoTrack.enabled);
  console.log("Track Muted:", videoTrack.muted);
  console.log("Track Content Hint:", videoTrack.contentHint);

  // Log track settings (actual current values)
  const settings = videoTrack.getSettings();
  console.log("CAMERA METADATA - Track Settings:", settings);

  // Log track capabilities (what the camera can do)
  const capabilities = videoTrack.getCapabilities();
  console.log("CAMERA METADATA - Track Capabilities:", capabilities);

  // Log track constraints (what was requested)
  const constraints = videoTrack.getConstraints();
  console.log("CAMERA METADATA - Track Constraints:", constraints);

  // Create metadata display
  const metadataDiv = document.createElement("div");
  metadataDiv.id = "cameraMetadata";
  metadataDiv.style.fontFamily = "monospace";
  metadataDiv.style.fontSize = "12px";
  metadataDiv.style.padding = "10px";
  metadataDiv.style.backgroundColor = "#f8f8f8";
  metadataDiv.style.border = "1px solid #ddd";
  metadataDiv.style.borderRadius = "4px";
  metadataDiv.style.margin = "10px 0";
  metadataDiv.style.maxHeight = "200px";
  metadataDiv.style.overflow = "auto";

  // Format metadata for display
  metadataDiv.innerHTML = `
    <h3>Camera Metadata</h3>
    <p><strong>Device:</strong> ${videoTrack.label}</p>
    <p><strong>Track ID:</strong> ${videoTrack.id}</p>
    <p><strong>Resolution:</strong> ${settings.width}×${settings.height}</p>
    <p><strong>Frame Rate:</strong> ${settings.frameRate || "Unknown"}</p>
    <p><strong>Aspect Ratio:</strong> ${settings.aspectRatio || "Unknown"}</p>
    <p><strong>Facing Mode:</strong> ${settings.facingMode || "Unknown"}</p>
    <p><strong>Resizable:</strong> ${capabilities && capabilities.width ? "Yes" : "No"}</p>
    <p><strong>White Balance Mode:</strong> ${settings.whiteBalanceMode || "Unknown"}</p>
    <p><strong>Exposure Mode:</strong> ${settings.exposureMode || "Unknown"}</p>
    <p><strong>Focus Mode:</strong> ${settings.focusMode || "Unknown"}</p>
  `;

  // Get or create video container
  let videoContainer = document.getElementById("videoContainer");
  if (!videoContainer) {
    videoContainer = document.createElement("div");
    videoContainer.id = "videoContainer";
    videoContainer.style.display = "flex";
    videoContainer.style.flexDirection = "row";
    videoContainer.style.flexWrap = "wrap";
    videoContainer.style.gap = "20px";
    videoContainer.style.marginTop = "20px";
    videoContainer.style.justifyContent = "center";
    videoContainer.style.maxWidth = "1440px";
    videoContainer.style.margin = "20px auto";
    document.body.appendChild(videoContainer);
  }

  // Create wrapper for video feed and processed frames
  const visualsWrapper = document.createElement("div");
  visualsWrapper.style.display = "flex";
  visualsWrapper.style.flexDirection = "column";
  visualsWrapper.style.gap = "20px";
  visualsWrapper.style.flex = "1";
  visualsWrapper.style.minWidth = "300px";
  visualsWrapper.style.maxWidth = "100%";

  // Create canvas once
  const canvas = document.createElement("canvas");
  canvas.width = settings.width || 640;
  canvas.height = settings.height || 480;
  canvas.style.borderRadius = "8px";
  canvas.style.border = "1px solid #ddd";
  canvas.style.maxWidth = "100%";
  canvas.style.height = "auto";

  // Create canvas section
  const canvasSection = document.createElement("div");
  canvasSection.style.marginBottom = "10px";

  const canvasLabel = document.createElement("div");
  canvasLabel.textContent = "Processed Frames";
  canvasLabel.style.fontWeight = "bold";
  canvasLabel.style.marginBottom = "5px";
  canvasSection.appendChild(canvasLabel);

  // Remove existing canvas if present
  const existingCanvas = document.getElementById("captureCanvas");
  if (existingCanvas) {
    existingCanvas.parentElement.remove();
  }

  canvas.id = "captureCanvas";
  canvasSection.appendChild(canvas);

  // Create metadata section
  const metadataSection = document.createElement("div");
  metadataSection.style.flex = "1";
  metadataSection.style.minWidth = "300px";
  metadataSection.style.maxWidth = "500px";
  metadataSection.style.alignSelf = "flex-start";

  // Remove existing metadata if present
  const existingMetadata = document.getElementById("cameraMetadata");
  if (existingMetadata) {
    existingMetadata.remove();
  }

  metadataSection.appendChild(metadataDiv);

  // Add canvas to the visuals wrapper
  visualsWrapper.appendChild(canvasSection);

  // Add sections to container
  videoContainer.appendChild(visualsWrapper);
  videoContainer.appendChild(metadataSection);

  // Create ImageCapture instance once
  const imageCapture = new ImageCapture(videoTrack);

  // Try to get and log photo capabilities if available
  try {
    const photoCapabilities = await imageCapture.getPhotoCapabilities();
    console.log("CAMERA METADATA - Photo Capabilities:", photoCapabilities);

    // Add to metadata display
    const photoCapDiv = document.createElement("div");
    photoCapDiv.innerHTML = `
      <h4>Photo Capabilities</h4>
      <p><strong>Red Eye Reduction:</strong> ${photoCapabilities.redEyeReduction || "Not available"}</p>
      <p><strong>Image Height:</strong> ${photoCapabilities.imageHeight?.min || "Unknown"} - ${photoCapabilities.imageHeight?.max || "Unknown"}</p>
      <p><strong>Image Width:</strong> ${photoCapabilities.imageWidth?.min || "Unknown"} - ${photoCapabilities.imageWidth?.max || "Unknown"}</p>
      <p><strong>Fill Light Mode:</strong> ${Array.isArray(photoCapabilities.fillLightMode) ? photoCapabilities.fillLightMode.join(", ") : "Not available"}</p>
    `;
    metadataDiv.appendChild(photoCapDiv);
  } catch (err) {
    console.log("Photo capabilities not available:", err.message);
  }

  // Get context once
  const ctx = canvas.getContext("bitmaprenderer") || canvas.getContext("2d");
  if (!ctx) {
    console.error("Could not get canvas context");
    return;
  }

  let animationId;
  let isRunning = true;
  let frameCount = 0;
  let lastFrameTime = performance.now();
  let actualFps = 0;

  // Function to capture and render each frame
  async function captureAndRenderFrame() {
    if (!isRunning || videoTrack.readyState !== "live") {
      console.log("Track is no longer live, stopping capture");
      return;
    }

    try {
      // Capture new frame
      const imageBitmap = await imageCapture.grabFrame();

      // Calculate actual FPS
      const now = performance.now();
      frameCount++;
      if (now - lastFrameTime >= 1000) {
        actualFps = frameCount;
        frameCount = 0;
        lastFrameTime = now;

        // Update fps in metadata
        const fpsElement = metadataDiv.querySelector(".actual-fps");
        if (fpsElement) {
          fpsElement.textContent = actualFps;
        } else {
          const fpsInfo = document.createElement("p");
          fpsInfo.innerHTML = `<strong>Actual FPS:</strong> <span class="actual-fps">${actualFps}</span>`;
          metadataDiv.insertBefore(fpsInfo, metadataDiv.children[2]);
        }
      }

      // Render to canvas
      if (ctx instanceof ImageBitmapRenderingContext) {
        ctx.transferFromImageBitmap(imageBitmap);
      } else if (ctx instanceof CanvasRenderingContext2D) {
        ctx.drawImage(imageBitmap, 0, 0);
      }

      // Clean up bitmap to prevent memory leaks
      imageBitmap.close();

      // Request next frame
      animationId = requestAnimationFrame(captureAndRenderFrame);
    } catch (error) {
      console.error("Frame capture error:", error);
      if (videoTrack.readyState !== "ended") {
        // If not ended, try again
        animationId = requestAnimationFrame(captureAndRenderFrame);
      }
    }
  }

  // Start the capture loop
  captureAndRenderFrame();

  // Add stop method to document for testing
  document.stopCapture = () => {
    isRunning = false;
    if (animationId) {
      cancelAnimationFrame(animationId);
    }
    console.log("Frame capture stopped");
  };

  console.log(
    "Continuous capture started. Call document.stopCapture() to stop.",
  );
}

/**
 * Creates and manages a camera selection interface
 * @returns {Promise<void>}
 */
async function setupCameraChooser() {
  // Create UI container with styling
  const container = document.createElement("div");
  container.style.margin = "10px 0";
  container.style.padding = "10px";
  container.style.backgroundColor = "#f0f0f0";
  container.style.borderRadius = "5px";

  // Create camera selector
  const cameraSelect = document.createElement("select");
  const label = document.createElement("label");
  label.textContent = "Select camera: ";
  label.htmlFor = "camera-select";
  cameraSelect.id = "camera-select";
  cameraSelect.style.margin = "0 10px";
  cameraSelect.style.padding = "5px";

  // Add refresh button
  const refreshButton = document.createElement("button");
  refreshButton.textContent = "🔄 Refresh";
  refreshButton.style.padding = "5px 10px";
  refreshButton.style.marginLeft = "10px";

  // Add elements to container
  container.appendChild(label);
  container.appendChild(cameraSelect);
  container.appendChild(refreshButton);

  // Add to document
  document.body.insertBefore(container, document.body.firstChild);

  let currentStream = null;

  // Function to populate the camera list
  async function populateCameraList() {
    try {
      // Clear existing options
      cameraSelect.innerHTML = "";

      // Get all media devices
      const devices = await navigator.mediaDevices.enumerateDevices();
      console.log("DEVICE METADATA - All Media Devices:", devices);

      // Filter for video inputs (cameras)
      const videoDevices = devices.filter(
        (device) => device.kind === "videoinput",
      );
      console.log("DEVICE METADATA - Video Input Devices:", videoDevices);

      if (videoDevices.length === 0) {
        const option = document.createElement("option");
        option.text = "No cameras found";
        cameraSelect.add(option);
        cameraSelect.disabled = true;
        return false;
      }

      // Add each camera to the select
      videoDevices.forEach((device) => {
        const option = document.createElement("option");
        option.value = device.deviceId;
        option.text =
          device.label || `Camera ${cameraSelect.options.length + 1}`;
        cameraSelect.add(option);
      });

      // Enable the select
      cameraSelect.disabled = false;
      return true;
    } catch (err) {
      console.error("Error enumerating devices:", err);
      const option = document.createElement("option");
      option.text = "Error getting cameras";
      cameraSelect.add(option);
      cameraSelect.disabled = true;
      return false;
    }
  }

  // Function to start the selected camera
  async function startSelectedCamera() {
    // Stop any existing capture
    if (document.stopCapture) {
      document.stopCapture();
    }

    // Stop any existing stream
    if (currentStream) {
      currentStream.getTracks().forEach((track) => track.stop());
    }

    const deviceId = cameraSelect.value;
    console.log("Starting camera with deviceId:", deviceId);

    try {
      // Try to get the device's capabilities to request optimal resolution
      const constraints = {
        video: {
          deviceId: deviceId ? { exact: deviceId } : undefined,
          // Request high resolution if available
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          frameRate: { ideal: 30 },
        },
        audio: false, // No need for audio for frame capture
      };

      console.log("Using constraints:", constraints);

      // Start a new stream with the selected camera
      currentStream = await navigator.mediaDevices.getUserMedia(constraints);
      console.log("Stream obtained:", currentStream);

      const videoTrack = currentStream.getVideoTracks()[0];
      console.log("Selected video track:", videoTrack);

      // Get or create container for video display
      let videoContainer = document.getElementById("videoContainer");
      if (!videoContainer) {
        // Create video container
        videoContainer = document.createElement("div");
        videoContainer.id = "videoContainer";
        videoContainer.style.display = "flex";
        videoContainer.style.flexDirection = "row";
        videoContainer.style.flexWrap = "wrap";
        videoContainer.style.gap = "20px";
        videoContainer.style.marginTop = "20px";
        document.body.appendChild(videoContainer);
      }

      // Create or get video element
      let videoElement = document.getElementById("liveVideo");
      let videoSection = document.getElementById("videoSection");

      if (!videoElement) {
        // Create video section
        videoSection = document.createElement("div");
        videoSection.id = "videoSection";

        // Create video element
        videoElement = document.createElement("video");
        videoElement.id = "liveVideo";
        videoElement.autoplay = true;
        videoElement.playsInline = true;
        videoElement.muted = true;
        videoElement.style.width = "480px";
        videoElement.style.height = "360px";
        videoElement.style.backgroundColor = "#000";
        videoElement.style.borderRadius = "8px";
        videoElement.style.border = "1px solid #ddd";

        // Create label for video
        const videoLabel = document.createElement("div");
        videoLabel.textContent = "Live Camera Feed";
        videoLabel.style.fontWeight = "bold";
        videoLabel.style.marginBottom = "5px";

        // Add elements to video section
        videoSection.appendChild(videoLabel);
        videoSection.appendChild(videoElement);

        // Add to container (at the beginning)
        if (videoContainer.firstChild) {
          videoContainer.insertBefore(videoSection, videoContainer.firstChild);
        } else {
          videoContainer.appendChild(videoSection);
        }
      }

      // Connect stream to video element
      videoElement.srcObject = currentStream;

      // Start the frame capture with the new track
      if (videoTrack) {
        exampleUsageImageCapture(videoTrack);
      }
    } catch (err) {
      console.error("Error starting camera:", err);
      alert(`Failed to start camera: ${err.message}`);
    }
  }

  // Set up event listeners
  cameraSelect.addEventListener("change", startSelectedCamera);
  refreshButton.addEventListener("click", async () => {
    // Re-enumerate devices to refresh the list
    const hasDevices = await populateCameraList();
    if (hasDevices) {
      startSelectedCamera();
    }
  });

  // Initial setup - get permission and populate camera list
  try {
    // Get initial camera access to prompt for permission
    currentStream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    // Now we can enumerate devices with labels
    const hasDevices = await populateCameraList();

    if (hasDevices) {
      // Set dropdown to match current device if possible
      const currentTrack = currentStream.getVideoTracks()[0];
      if (currentTrack) {
        const currentSettings = currentTrack.getSettings();
        if (currentSettings.deviceId) {
          cameraSelect.value = currentSettings.deviceId;
        }
      }

      // Start capture with the selected (or default) camera
      startSelectedCamera();
    }
  } catch (err) {
    console.error("Error accessing camera:", err);
    alert(
      `Camera access error: ${err.message}\nPlease ensure your camera is connected and you've granted permission.`,
    );
  }
}

// Initialize the camera chooser instead of directly accessing the camera
document.addEventListener("DOMContentLoaded", setupCameraChooser);
