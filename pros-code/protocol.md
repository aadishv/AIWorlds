# Serial Communication Protocol

This document describes the serial communication protocol used between the PROS robot controller and an external vision/detection system.

## Overview

The protocol enables bidirectional communication using JSON-formatted messages:

1. The PROS robot sends its current pose information (position and orientation) to the external system
2. The external system responds with detection data (objects identified in the environment)

The communication happens over serial connection with COBS encoding disabled.

## Messages From Robot to External System

The robot sends pose information in the following JSON format:

```json
{
    "x": 1.0,      // X coordinate (double)
    "y": 2.0,      // Y coordinate (double)
    "theta": 180.0 // Orientation in degrees (double)
}
```

Each message is terminated with `\r\n` (CRLF) and is followed by a flush of the output buffer.

## Messages From External System to Robot

The external system sends detection information in the following JSON format:

```json
{
    "flag": "optional_flag_string",  // Optional flag field
    "pose": {
        "x": 0.0,      // X coordinate (double)
        "y": 0.0,      // Y coordinate (double)
        "theta": 0.0   // Orientation in degrees (double)
    },
    "stuff": [         // Array of detected objects
        {
            "x": 0.0,       // X coordinate of detection (double)
            "y": 0.0,       // Y coordinate of detection (double)
            "z": 0.0,       // Z coordinate of detection (double)
            "class": "type" // Class/type of the detected object (string, one of: "red", "blue", "bot", "goal")
        },
        // Additional detections...
    ]
}
```

The "pose" field is required for a valid message. The "stuff" array contains the list of detected objects, each with position coordinates and a classification.

## Communication Flow

1. **Initialization**:
   - The robot calls `serial::start()` which disables COBS encoding and sends the initial pose
   - Initial pose values may be configured from autonomous selector settings

2. **Regular Operation**:
   - The robot periodically sends updated pose information (every 20ms)
   - The robot attempts to read and parse incoming JSON messages
   - Successfully parsed messages update the global `most_recent_frame` variable
   - The UI displays the number of current detections

## Error Handling

The protocol implements robust error handling:

1. Invalid JSON messages are logged with error details
2. Parse errors (invalid JSON structure) are caught and reported
3. When messages cannot be parsed, the system returns a null optional (`std::nullopt`)
4. The UI defaults to showing "Detections: 0" when no valid frame is available

## Implementation Notes

- Parsing is done using the nlohmann/json library
- All communication values are passed through the standard input/output streams
- The `serctl` function is used to disable COBS encoding for proper serial communication
- The current implementation sends fixed pose data (1.0, 2.0, 180.0) but should be updated to use actual IMU readings

## Example Communication

**Robot to External System**:
```
{"x": 1.0,"y": 2.0, "theta": 180.0}
```

**External System to Robot**:
```
{"flag": "tracking", "pose": {"x": 1.5, "y": 2.3, "theta": 175.0}, "stuff": [{"x": 3.2, "y": 1.7, "z": 0.5, "class": "goal"}, {"x": 5.1, "y": 4.2, "z": 0.0, "class": "red"}]}
```

## Future Improvements

1. Update the implementation to use actual IMU readings instead of fixed values
2. Consider adding timestamps to messages for synchronization
3. Add more detailed error codes or status flags
4. Implement bidirectional acknowledgment of received messages