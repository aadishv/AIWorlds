import atexit
import sys
import threading

import serial
from serial.tools import list_ports


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
    ports = list_ports.comports()
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
