import threading
import sys
import serial
from serial.tools import list_ports


class EventDrivenSerial:
    def __init__(self,
                 port,
                 baudrate,
                 callback,
                 eol=b'\n',
                 timeout=1.0,
                 reconnect_delay=5.0,
                 log_filename='serial_log.txt'):
        """
        :param port:            Serial port name, e.g. 'COM3' or '/dev/ttyUSB0'
        :param baudrate:        Baud rate, e.g. 9600
        :param callback:        Function f(line:str) -> Optional[str]
                                Called on each received line (without newline).
                                If it returns a string, that string (with newline)
                                is written back to the serial port.
        :param eol:             Byte sequence for line endings (default b'\\n')
        :param timeout:         Serial read timeout in seconds
        :param reconnect_delay: Seconds to wait before attempting reconnection after an error
        :param log_filename:    File to which we append all RX/TX traffic
        """
        self.port = port
        self.baudrate = baudrate
        self.callback = callback
        self.eol = eol
        self.timeout = timeout
        self.reconnect_delay = reconnect_delay

        self._ser = None
        self._lock = threading.Lock()       # protects serial writes
        self._log_lock = threading.Lock()   # protects log file writes
        self._stop_event = threading.Event()
        self._thread = None

        # Open the log file in append mode once
        self._log_file = open(log_filename, 'a', buffering=1, encoding='utf-8')
        # buffering=1 means line-buffered. encoding=utf-8 so we're safe with unicode.

    def _log(self, direction, line):
        """Internal helper to write a timestamped RX/TX line to the log."""
        from datetime import datetime
        timestamp = datetime.now().isoformat(sep=' ', timespec='milliseconds')
        entry = f"{timestamp} {direction}: {line}\n"
        with self._log_lock:
            self._log_file.write(entry)

    def _connect(self):
        """Attempts to connect to the serial port."""
        if self._ser and self._ser.is_open:
            return True
        try:
            self._close_serial()
            print(
                f"[EventDrivenSerial] Attempting to connect to {self.port}...")
            temp_timeout = min(self.timeout, 1.0)
            self._ser = serial.Serial(self.port, self.baudrate,
                                      timeout=temp_timeout)
            self._ser.timeout = self.timeout
            print(f"[EventDrivenSerial] Connected to {self.port}")
            return True
        except serial.SerialException as e:
            print(f"[EventDrivenSerial] Connection failed: {e}")
            self._ser = None
            return False
        except Exception as e:
            print(
                f"[EventDrivenSerial] Unexpected error during connection: {e}")
            self._ser = None
            return False

    def _close_serial(self):
        if self._ser and self._ser.is_open:
            try:
                self._ser.close()
                print(f"[EventDrivenSerial] Port {self.port} closed.")
            except Exception as e:
                print(
                    f"[EventDrivenSerial] Error closing port {self.port}: {e}")
        self._ser = None

    def start(self):
        if self._thread and self._thread.is_alive():
            print("[EventDrivenSerial] Worker thread already running.")
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self.serial_worker,
                                        daemon=True)
        self._thread.start()
        print("[EventDrivenSerial] Worker thread started.")

    def stop(self):
        if not self._thread or not self._thread.is_alive():
            print("[EventDrivenSerial] Worker thread not running.")
            return
        print("[EventDrivenSerial] Stopping worker thread...")
        self._stop_event.set()

    def serial_worker(self):
        import time
        PERIOD = 1.0 / 60.0
        last_time = time.perf_counter() - PERIOD

        while not self._stop_event.is_set():
            # if not connected, try to reconnect
            if not (self._ser and self._ser.is_open):
                if not self._connect():
                    if self._stop_event.wait(self.reconnect_delay):
                        break
                    continue

            try:
                while not self._stop_event.is_set():
                    # now = time.perf_counter()
                    # elapsed = now - last_time
                    # if elapsed < PERIOD:
                    #     if self._stop_event.wait(PERIOD - elapsed):
                    #         break

                    raw = self._ser.readline()
                    if self._stop_event.is_set():
                        break
                    if not raw:
                        continue

                    # RX: decode + strip
                    line = raw.decode('utf-8', errors='replace').rstrip('\r\n')
                    print(f"RX<< {line}")
                    self._log('RX', line)

                    # run callback
                    try:
                        response = self.callback(line)
                    except Exception as cb_err:
                        print(f"[EventDrivenSerial] callback error: {cb_err}")
                        response = None

                    if isinstance(response, str):
                        # TX: strip any stray newlines then re-append your eol
                        out_line = response.rstrip('\r\n')
                        print(f"TX>> {out_line}")
                        self._log('TX', out_line)

                        out = out_line.encode('utf-8') + self.eol
                        with self._lock:
                            if not (self._ser and self._ser.is_open):
                                raise serial.SerialException(
                                    "Lost connection before write")
                            self._ser.write(out)

                    # last_time = time.perf_counter()

            except serial.SerialException as ser_err:
                print(
                    f"[EventDrivenSerial] Serial error: {ser_err}. Disconnected.")
                self._close_serial()
                if self._stop_event.wait(1.0):
                    break
            except Exception as e:
                print(f"[EventDrivenSerial] Unexpected error: {e}")
                self._close_serial()
                if self._stop_event.wait(self.reconnect_delay):
                    break

        # cleanup
        print("[EventDrivenSerial] Worker thread loop finished.")
        self._close_serial()
        # close the log file
        with self._log_lock:
            self._log_file.close()
        print("[EventDrivenSerial] Worker thread exiting.")

# helper functions unchanged


def list_serial_ports():
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
