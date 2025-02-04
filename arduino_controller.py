import os
import serial
import time
import platform
from serial.tools import list_ports

class ArduinoController:
    def __init__(self):
        self.arduino = None
        try:
            if "microsoft" in platform.uname().release.lower():
                port = '/dev/ttyS2'  # Update if necessary
                if not os.path.exists(port):
                    raise Exception(f"Port {port} does not exist. {self.usbip_instructions()}")
            else:
                port = 'COM3' if platform.system() == 'Windows' else '/dev/ttyACM0'
            
            self.arduino = serial.Serial(port, 4800, timeout=1)
            time.sleep(2)
        except Exception as e:
            error_message = str(e)
            error_message += f" | Available ports: {self.get_available_ports()}"
            print(f"Could not connect to Arduino: {error_message}")
            self.arduino = None

    def get_available_ports(self):
        ports = [port.device for port in list_ports.comports()]
        return ports

    def usbip_instructions(self):
        return ("No serial ports found. If you're using WSL2, try attaching your Arduino with USBIP. "
                "In WSL, run 'usbipd wsl list' to list USB devices and then "
                "'sudo usbipd wsl attach --busid <busid>' to attach your Arduino, "
                "after which the device should appear (e.g., as /dev/ttyS2).")

    def set_led(self, is_condition_detected):
        if self.arduino:
            self.arduino.write(b'1' if is_condition_detected else b'0')

    def close(self):
        if self.arduino:
            self.arduino.close()
