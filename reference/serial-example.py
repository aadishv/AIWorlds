#!/usr/bin/env python3.6

import json
import random
import sys
import threading
import time

from schema import Detection
from serial_impl import SerialPortManager, list_serial_ports, prompt_user_for_port

# Types of messages
# Message type A: Detected object (8). [cx, cy, w, h, cls_index, confidence, depth]
# Message type B: Localization readings (20). [depth, offset] * 10.
# Message type C: New frame indicator (0).


def main():
    # 1) enumerate & select
    ports = list_serial_ports()
    if not ports:
        print("No serial ports found.")
        sys.exit(1)
    port_name = prompt_user_for_port(ports)

    BAUD = 9600

    # These will hold our send‐loop thread & stop event
    send_thread = None
    send_stop_evt = threading.Event()

    def on_receive(line):
        nonlocal send_thread
        print(f"[Reader] Received: {line}")
        if "activate" in line and send_thread is None:
            print("[Reader] Activation detected, starting send loop.")
            send_thread = threading.Thread(
                target=send_loop, name="SenderLoop", daemon=True
            )
            send_thread.start()

    def process_msg(t, l):
        return f"{t} {' '.join(list(map(str, l)))}"

    def send_loop():
        """
        Runs in its own thread once we get 'activate'.
        Sends Message1, Message2, ... every 1/30 second
        until send_stop_evt is set.
        """
        while not send_stop_evt.is_set():
            # mock data
            frame = {
                "points": [random.random() * 100 for _ in range(20)],
                "dets": [Detection.random().serialize() for _ in range(random.randint(2, 5))],
            }
            spm.send_message(json.dumps(frame))
            time.sleep(1 / 30.0)

    # 2) open the manager
    print(f"Opening {port_name} @ {BAUD} baud …")
    spm = SerialPortManager(
        port=port_name, baudrate=BAUD, on_receive=on_receive, eol="\n", timeout=None
    )

    # 3) just sit here until CTRL+C
    try:
        print("Waiting for data / 'activate' …")
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nInterrupted by user, shutting down …")
    finally:
        # stop the send loop if running
        send_stop_evt.set()
        if send_thread:
            send_thread.join(timeout=1.0)

        # stop the serial manager
        spm.stop()
        print("Clean exit.")


if __name__ == "__main__":
    main()
