// app.js
const canvas = document.getElementById('overlay-canvas');
const ctx = canvas.getContext('2d');
const errorMsg = document.getElementById('error-message');
const instructions = document.getElementById('instructions');
const resetBtn = document.getElementById('reset-btn');
const debugSlices = document.getElementById('debug-slices');
const width = canvas.width;
const height = canvas.height;
let points = [];
let lastSliceDebug = [];
let videoSource = null; // Will be set to either video or img

// Try webcam first, fallback to MJPEG img
function setupVideoSource() {
  const video = document.getElementById('video-stream');
  const img = document.getElementById('img-stream');
  if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
    navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 360 } })
      .then(stream => {
        video.srcObject = stream;
        video.style.display = '';
        img.style.display = 'none';
        videoSource = video;
        errorMsg.textContent = "";
      })
      .catch(err => {
        // Fallback to MJPEG stream
        video.style.display = 'none';
        img.style.display = '';
        videoSource = img;
        errorMsg.textContent = "";
      });
  } else {
    // Fallback to MJPEG stream
    video.style.display = 'none';
    img.style.display = '';
    videoSource = img;
    errorMsg.textContent = "";
  }
}

function saveToLocalStorage() {
  if (points.length === 2) {
    localStorage.setItem('segmentPoint1', JSON.stringify(points[0]));
    localStorage.setItem('segmentPoint2', JSON.stringify(points[1]));
  }
}
function loadFromLocalStorage() {
  let p1 = localStorage.getItem('segmentPoint1');
  let p2 = localStorage.getItem('segmentPoint2');
  if (p1 && p2) {
    try {
      points = [JSON.parse(p1), JSON.parse(p2)];
    } catch {
      points = [];
    }
  }
}
function clearLocalStorage() {
  localStorage.removeItem('segmentPoint1');
  localStorage.removeItem('segmentPoint2');
}

function colorDistance(rgb, ref) {
  return Math.sqrt(
    Math.pow(rgb[0] - ref[0], 2) +
    Math.pow(rgb[1] - ref[1], 2) +
    Math.pow(rgb[2] - ref[2], 2)
  );
}
function median(arr) {
  arr = arr.slice().sort((a, b) => a - b);
  const mid = Math.floor(arr.length / 2);
  return arr.length % 2 !== 0 ? arr[mid] : (arr[mid - 1] + arr[mid]) / 2;
}
function robustColor(pixels) {
  if (pixels.length === 0) return [127,127,127];
  let rs = pixels.map(p => p[0]);
  let gs = pixels.map(p => p[1]);
  let bs = pixels.map(p => p[2]);
  return [median(rs), median(gs), median(bs)];
}
function classifyColor(rgb) {
  const red = [220, 40, 40];
  const blue = [40, 80, 200];
  const threshold = 170; // further increased
  if (colorDistance(rgb, red) < threshold) return "red";
  if (colorDistance(rgb, blue) < threshold) return "blue";
  return "unknown";
}
function rgbToHex(rgb) {
  return "#" + rgb.map(x => {
    let h = x.toString(16);
    return h.length === 1 ? "0" + h : h;
  }).join('');
}
function drawDebugView(sliceDebug) {
  debugSlices.innerHTML = "";
  for (let i = 0; i < sliceDebug.length; ++i) {
    const {colorType, rgb} = sliceDebug[i];
    let swatchColor;
    let label;
    if (colorType === "red") {
      swatchColor = "var(--slice-red)";
      label = "Red";
    } else if (colorType === "blue") {
      swatchColor = "var(--slice-blue)";
      label = "Blue";
    } else if (colorType === "grey") {
      swatchColor = "var(--slice-grey)";
      label = "Grey";
    } else {
      swatchColor = "var(--slice-unknown)";
      label = "Unknown";
    }
    let hex = rgbToHex(rgb.map(x => Math.round(x)));
    debugSlices.innerHTML += `
      <div class="debug-slice">
        <div class="debug-swatch" style="background:${swatchColor};"></div>
        <div class="debug-label">${label}</div>
        <div style="font-size:0.82rem;color:#64748b;">${hex}</div>
      </div>
    `;
  }
}

function draw() {
  ctx.clearRect(0, 0, width, height);
  errorMsg.textContent = "";
  lastSliceDebug = [];

  if (points.length === 1) {
    ctx.fillStyle = "var(--slice-red)";
    ctx.beginPath();
    ctx.arc(points[0].x, points[0].y, 6, 0, 2 * Math.PI);
    ctx.fill();
  }
  if (points.length === 2) {
    let {x: x1, y: y1} = points[0];
    let {x: x2, y: y2} = points[1];

    ctx.fillStyle = "var(--slice-red)";
    ctx.beginPath();
    ctx.arc(x1, y1, 6, 0, 2 * Math.PI);
    ctx.arc(x2, y2, 6, 0, 2 * Math.PI);
    ctx.fill();

    let dx = x2 - x1;
    let dy = y2 - y1;
    let segLen = Math.sqrt(dx*dx + dy*dy);
    if (segLen > 1e-3 && videoSource) {
      let perp_dx = -dy / segLen;
      let perp_dy = dx / segLen;
      let rectWidth = 20/29 * segLen;
      let halfWidth = rectWidth / 2;
      let numSlices = 6;

      // Get video frame pixels
      let tempCanvas = document.createElement('canvas');
      tempCanvas.width = width;
      tempCanvas.height = height;
      let tempCtx = tempCanvas.getContext('2d');
      try {
        tempCtx.drawImage(videoSource, 0, 0, width, height);
      } catch (e) {
        // If drawImage fails, fallback to transparent
      }
      let frame;
      try {
        frame = tempCtx.getImageData(0, 0, width, height);
      } catch (e) {
        // If getImageData fails, fallback to transparent
        frame = {data: new Uint8ClampedArray(width*height*4)};
      }

      for (let i = 0; i < numSlices; ++i) {
        let t0 = i / numSlices;
        let t1 = (i + 1) / numSlices;

        let mx0 = x1 + dx * t0;
        let my0 = y1 + dy * t0;
        let mx1 = x1 + dx * t1;
        let my1 = y1 + dy * t1;

        let p1x = mx0 + perp_dx * halfWidth;
        let p1y = my0 + perp_dy * halfWidth;
        let p2x = mx1 + perp_dx * halfWidth;
        let p2y = my1 + perp_dy * halfWidth;
        let p3x = mx1 - perp_dx * halfWidth;
        let p3y = my1 - perp_dy * halfWidth;
        let p4x = mx0 - perp_dx * halfWidth;
        let p4y = my0 - perp_dy * halfWidth;

        // Build mask and collect pixels inside polygon
        let pixels = [];
        let minX = Math.floor(Math.min(p1x, p2x, p3x, p4x));
        let maxX = Math.ceil(Math.max(p1x, p2x, p3x, p4x));
        let minY = Math.floor(Math.min(p1y, p2y, p3y, p4y));
        let maxY = Math.ceil(Math.max(p1y, p2y, p3y, p4y));
        function pointInPoly(px, py) {
          let poly = [[p1x,p1y],[p2x,p2y],[p3x,p3y],[p4x,p4y]];
          let inside = false;
          for (let j = 0, k = poly.length - 1; j < poly.length; k = j++) {
            let xi = poly[j][0], yi = poly[j][1];
            let xj = poly[k][0], yj = poly[k][1];
            let intersect = ((yi > py) !== (yj > py)) &&
              (px < (xj - xi) * (py - yi) / (yj - yi + 1e-10) + xi);
            if (intersect) inside = !inside;
          }
          return inside;
        }
        for (let y = minY; y <= maxY; ++y) {
          for (let x = minX; x <= maxX; ++x) {
            if (x < 0 || x >= width || y < 0 || y >= height) continue;
            if (pointInPoly(x, y)) {
              let idx = (y * width + x) * 4;
              let r = frame.data[idx], g = frame.data[idx+1], b = frame.data[idx+2];
              pixels.push([r,g,b]);
            }
          }
        }
        let med = robustColor(pixels);
        let colorType = classifyColor(med);

        let fill;
        if (colorType === "red" || colorType === "blue") {
          fill = rgbToHex(med); // Use the actual median color for the fill
        } else if (colorType === "grey") {
          fill = "var(--slice-grey)";
        } else {
          fill = "var(--slice-unknown)";
        }

        lastSliceDebug.push({colorType, rgb: med});

        ctx.save();
        ctx.fillStyle = fill;
        ctx.beginPath();
        ctx.moveTo(p1x, p1y);
        ctx.lineTo(p2x, p2y);
        ctx.lineTo(p3x, p3y);
        ctx.lineTo(p4x, p4y);
        ctx.closePath();
        ctx.fill();
        ctx.restore();

        // Draw midline for this slice
        ctx.save();
        ctx.strokeStyle = "#fff";
        ctx.lineWidth = 2.2;
        ctx.beginPath();
        ctx.moveTo(mx0, my0);
        ctx.lineTo(mx1, my1);
        ctx.stroke();
        ctx.restore();
      }
    }
  }
  drawDebugView(lastSliceDebug);
}

canvas.addEventListener('click', function(e) {
  if (points.length === 2) return;
  const rect = canvas.getBoundingClientRect();
  const x = Math.round((e.clientX - rect.left) * (canvas.width / rect.width));
  const y = Math.round((e.clientY - rect.top) * (canvas.height / rect.height));
  points.push({x, y});
  if (points.length === 2) {
    saveToLocalStorage();
    instructions.textContent = "Segment defined. Click Reset to pick new points.";
  } else {
    instructions.textContent = "Click the second point on the video.";
  }
  draw();
});

resetBtn.addEventListener('click', function() {
  points = [];
  clearLocalStorage();
  instructions.textContent = "Click two points on the video to define the segment.";
  draw();
});

window.addEventListener('DOMContentLoaded', () => {
  setupVideoSource();
  loadFromLocalStorage();
  if (points.length === 2) {
    instructions.textContent = "Segment defined. Click Reset to pick new points.";
  } else {
    instructions.textContent = "Click two points on the video to define the segment.";
  }
  draw();
});
