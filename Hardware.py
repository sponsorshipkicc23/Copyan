import time
import cv2 as cv

SERIAL_AVAILABLE = False
esp_serial = None

try:
    import serial

    for _port in ("/dev/ttyUSB0", "/dev/ttyACM0"):
        try:
            esp_serial = serial.Serial(_port, 115200, timeout=1)
            SERIAL_AVAILABLE = True
            print(f"✅ ESP32 connected via {_port}")
            time.sleep(2)
            break
        except Exception:
            continue

    if not SERIAL_AVAILABLE:
        print("❌ ESP32 not detected. Running in Motor Simulation Mode.")

except ImportError:
    print("⚠️  pyserial not installed. Motor control disabled.")


def send_motor_command(direction: str, steps: int) -> None:
    cmd = f"{direction}{steps}\n"
    if SERIAL_AVAILABLE and esp_serial and esp_serial.is_open:
        try:
            esp_serial.write(cmd.encode("utf-8"))
            print(f"📡 → ESP32: {cmd.strip()}")
        except Exception as e:
            print(f"❌ Serial send failed: {e}")
    else:
        print(f"⚠️  Simulation – Motor {direction} {steps} steps")


def close_serial() -> None:
    if esp_serial and esp_serial.is_open:
        esp_serial.close()


# Camera Configuration
PICAM_AVAILABLE = False
QPicamera2 = None
Picamera2 = None

try:
    from picamera2.previews.qt import QPicamera2
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except Exception:
    pass


def open_webcam(index: int = 0) -> cv.VideoCapture:
    cap = cv.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open webcam at index {index}")
    return cap
