import threading
import serial
import sys

class EventDrivenSerial:
     def __init__(self, port, baudrate, callback, eol=b'\n', timeout=1.0):
        """
        :param port:      Serial port name, e.g. 'COM3' or '/dev/ttyUSB0'
        :param baudrate:  Baud rate, e.g. 9600
        :param callback:  Function f(line:str) -> Optional[str]
                          Called on each received line (without newline).
                          If it returns a string, that string (with newline)
                          is written back to the serial port.
        :param eol:       Byte sequence for line endings (default b'\\n')
        :param timeout:   Serial read timeout in seconds
        """
        self.port = port
        self.baudrate = baudrate
        self.callback = callback
        self.eol = eol
        self.timeout = timeout

        self._ser = None
        self._lock = threading.Lock()

        def serial_worker(self):
            try:
                self._ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            except serial.SerialException as e:
                raise RuntimeError(f"Could not open serial port {self.port}: {e}")

            try:
                while True:
                    try:
                        raw = self._ser.readline()
                        if not raw:
                            continue  # timeout, loop again

                        # decode and strip newline(s)
                        line = raw.decode('utf-8', errors='replace').rstrip('\r\n')

                        # call user callback
                        try:
                            response = self.callback(line)
                        except Exception as cb_err:
                            # swallow errors in callback to keep thread alive
                            print(f"[EventDrivenSerial] callback error: {cb_err}")
                            continue

                        # if callback returned a string, send it back
                        if isinstance(response, str):
                            out = response.rstrip('\r\n').encode('utf-8') + self.eol
                            with self._lock:
                                self._ser.write(out)

                    except serial.SerialException as ser_err:
                        print(f"[EventDrivenSerial] serial error: {ser_err}")
                        break
                    except Exception as e:
                        # catch‐all to avoid thread exit
                        print(f"[EventDrivenSerial] unexpected error: {e}")
                        continue
            finally:
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
