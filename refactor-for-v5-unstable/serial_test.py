#!/usr/bin/env python3
"""
wait_for_activate.py

1) List available serial ports
2) Prompt user to select one
3) Open the selected port
4) Block and read until an "activate" line is received
"""

import sys
import serial
import serial.tools.list_ports


def list_serial_ports():
    """Return a list of available serial ports on the system."""
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]


def prompt_user_for_port(ports):
    """Prompt the user to choose a port from the list."""
    print("Available serial ports:")
    for idx, port in enumerate(ports):
        print(f"  {idx+1}: {port}")
    choice = input(f"Select port [1-{len(ports)}]: ").strip()
    try:
        idx = int(choice) - 1
        if not (0 <= idx < len(ports)):
            raise ValueError()
    except ValueError:
        print("Invalid selection. Exiting.")
        sys.exit(1)
    return ports[idx]


def main():
    # 1) List ports
    ports = list_serial_ports()
    if not ports:
        print("No serial ports found.")
        sys.exit(1)

    # 2) Prompt for port
    port_name = prompt_user_for_port(ports)
    baudrate = 9600       # Change as needed
    timeout_seconds = None  # Block forever

    print(f"Opening {port_name} at {baudrate} baud...")
    try:
        ser = serial.Serial(port=port_name,
                            baudrate=baudrate,
                            timeout=timeout_seconds)
    except serial.SerialException as e:
        print(f"Could not open port {port_name}: {e}")
        sys.exit(1)

    print("Port opened. Waiting for 'activate'...")

    try:
        while True:
            # Read a line (ending in \n). Adjust eol or use read() if needed.
            raw = ser.readline()
            if not raw:
                # With timeout=None this shouldn't happen, but just in case
                continue
            try:
                line = raw.decode('utf-8', errors='replace').strip()
            except Exception:
                line = raw.strip()
            if line:
                print(f"Received: {line}")
            if line.lower() == "activate":
                print("** ACTIVATION COMMAND RECEIVED **")
                break

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    finally:
        print("Closing port...")
        ser.close()
        print("Done.")


if __name__ == "__main__":
    main()
