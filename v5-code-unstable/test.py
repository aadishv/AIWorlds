#!/usr/bin/env python3.6
import atexit
import random
import sys
import threading
import time

import serial
import serial.tools.list_ports


class SerialPortManager:
    """
    Manages a serial port in a background reader thread, and
    exposes send_message() for writing.

    You supply a callback that takes one argument (the decoded line).
    """

    def __init__(self, port, baudrate, on_receive, eol="\n", timeout=None):
        """
        port:      e.g. '/dev/ttyUSB0' or 'COM3'
        baudrate:  e.g. 9600
        on_receive: a function f(str_line)
        eol:       line ending to append when sending
        timeout:   passed to Serial (None = block forever)
        """
        self.port = port
        self.baudrate = baudrate
        self._on_receive = on_receive
        self._eol = eol.encode("utf-8")
        self._timeout = timeout

        self._lock = threading.Lock()
        self._stop_evt = threading.Event()
        self._reader_thread = threading.Thread(
            target=self._read_loop, name="SerialReader", daemon=True
        )

        try:
            self._ser = serial.Serial(
                port=self.port, baudrate=self.baudrate, timeout=self._timeout
            )
        except serial.SerialException as e:
            raise RuntimeError(f"Could not open {port}: {e}")

        # ensure cleanup on normal exit
        atexit.register(self.stop)

        # start the reader
        self._reader_thread.start()

    def _read_loop(self):
        """
        Runs in background, reads lines, calls the on_receive callback.
        """
        try:
            while not self._stop_evt.is_set():
                raw = self._ser.readline()
                if not raw:
                    # no data or timed out
                    continue
                try:
                    line = raw.decode("utf-8", errors="replace").strip()
                except Exception:
                    line = raw.strip()
                if line:
                    self._on_receive(line)
        except Exception as e:
            # you might log this
            print(f"[SerialReader] Exception: {e}")
        # thread will exit

    def send_message(self, msg):
        """
        Send a string (without EOL). EOL is appended automatically.
        Thread-safe.
        """
        with self._lock:
            data = msg.encode("utf-8") + self._eol
            self._ser.write(data)

    def stop(self):
        """
        Stop the reader thread and close the port.
        Safe to call multiple times.
        """
        if not self._stop_evt.is_set():
            self._stop_evt.set()
            if self._reader_thread.is_alive():
                self._reader_thread.join(timeout=1.0)
            try:
                self._ser.close()
            except Exception:
                pass


def list_serial_ports():
    """Return a list of available port device names."""
    ports = serial.tools.list_ports.comports()
    return [p.device for p in ports]


def prompt_user_for_port(ports):
    print("Available serial ports:")
    for idx, p in enumerate(ports, 1):
        print(f"  {idx}: {p}")
    choice = input(f"Select port [1-{len(ports)}]: ").strip()
    try:
        i = int(choice) - 1
        if i < 0 or i >= len(ports):
            raise ValueError
    except ValueError:
        print("Invalid selection. Exiting.")
        sys.exit(1)
    return ports[i]


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
        i = 1
        while not send_stop_evt.is_set():
            # start new frame
            spm.send_message(process_msg("c", []))
            if i % random.randint(2, 5) == 0:  # simulate 7.5 fps detection rate
                for j in range(random.randint(2, 5)):
                    ns = [random.random() * 100 for _ in range(8)]
                    spm.send_message(process_msg("a", ns))
            spm.send_message(
                process_msg("b", [random.random() * 100 for _ in range(20)])
            )
            i += 1
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
