#!/usr/bin/env python3
import cv2
import time

def check_camera(index):
    """Try to open a camera with the given index and return if it's available."""
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        print(f"Camera index {index} is not available")
        return False
    
    # Get camera resolution
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    # Try to read a frame
    ret, frame = cap.read()
    if not ret:
        print(f"Camera index {index} is available but cannot read frames")
        cap.release()
        return False
    
    print(f"Camera index {index} is available - Resolution: {int(width)}x{int(height)}")
    
    # Display the camera feed briefly
    window_name = f"Camera {index}"
    cv2.imshow(window_name, frame)
    cv2.waitKey(1500)  # Show for 1.5 seconds
    cv2.destroyWindow(window_name)
    
    cap.release()
    return True

def main():
    print("Checking for available cameras...")
    print("This may take a minute, please wait...")
    
    # Check a reasonable range of camera indices
    # Typically, built-in webcams are 0, USB cameras might be 1, 2, etc.
    available_cameras = []
    
    # First check common indices quickly
    for index in range(5):
        if check_camera(index):
            available_cameras.append(index)
    
    # If no cameras found in common indices, try extended range
    if not available_cameras:
        print("\nNo cameras found in common indices, checking extended range...")
        for index in range(5, 10):
            if check_camera(index):
                available_cameras.append(index)
    
    # Print summary
    print("\nSummary:")
    if available_cameras:
        print(f"Found {len(available_cameras)} available camera(s) at indices: {available_cameras}")
        print("\nUse these indices with the blob detection script:")
        for idx in available_cameras:
            print(f"python3.8 blob_detection.py --camera {idx}")
    else:
        print("No cameras found on your system.")
        print("You might need to:")
        print("1. Check if your camera is properly connected")
        print("2. Verify your camera permissions")
        print("3. Try a video file instead: python3.8 blob_detection.py --video path/to/video.mp4")

if __name__ == "__main__":
    main()