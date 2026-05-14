import serial
import time
import numpy as np
import cv2 as cv
from PIL import Image

try:
    from picamera2.previews.qt import QPicamera2
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except Exception:
    QPicamera2 = None
    Picamera2 = None
    PICAM_AVAILABLE = False

class ESP32Controller:
    def __init__(self):
        self.serial_conn = None
        self.available = False
        self._connect()

    def _connect(self):
        ports = ['/dev/ttyUSB0', '/dev/ttyACM0']
        for port in ports:
            try:
                self.serial_conn = serial.Serial(port, 115200, timeout=1)
                self.available = True
                print(f"✅ ESP32 Terhubung via {port}!")
                time.sleep(2)
                break
            except Exception:
                continue
        if not self.available:
            print("❌ ESP32 Tidak Terdeteksi. Program jalan dalam Mode Simulasi Motor.")

    def send_command(self, direction_char, steps):
        if self.available and self.serial_conn and self.serial_conn.is_open:
            pesan = f"{direction_char}{steps}\n"
            try:
                self.serial_conn.write(pesan.encode('utf-8'))
                print(f"📡 Mengirim ke ESP32: {pesan.strip()}")
            except Exception as e:
                print(f"❌ Gagal mengirim perintah serial: {e}")
        else:
            print(f"⚠️ Mode Simulasi: Motor {direction_char} {steps} langkah")

    def stop(self):
        if self.available and self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.write("S0\n".encode('utf-8'))
            print("🛑 Motor Stop")

    def close(self):
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()

class MagnificationSensor:
    def __init__(self):
        pass # Area inisialisasi sensor I2C/analog
        
    def read_distance(self):
        return 15.4 

class CameraSystem:
    def __init__(self):
        self.using_picam = PICAM_AVAILABLE
        self.picam2 = None
        self.qpicamera2 = None
        self.cap = None

        if self.using_picam:
            try:
                self.picam2 = Picamera2()
                self.picam2.configure(self.picam2.create_preview_configuration({"size": (480, 270)}))
                self.qpicamera2 = QPicamera2(self.picam2, width=480, height=270, keep_ar=True)
            except Exception:
                self.using_picam = False

        if not self.using_picam:
            self.cap = cv.VideoCapture(0)

    def start_camera(self):
        if self.using_picam and self.picam2:
            try: self.picam2.start()
            except Exception: pass

    def capture_image(self, save_path, signal_callback=None):
        if self.using_picam and self.picam2:
            cfg = self.picam2.create_still_configuration(main={"size": (480, 270)})
            self.picam2.switch_mode_and_capture_file(cfg, save_path, signal_function=signal_callback)
            return True
        elif self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                cv.imwrite(save_path, frame)
                return True
        return False

    def get_opencv_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret: return cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        return None

    def close(self):
        if self.using_picam and self.picam2:
            try: self.picam2.stop()
            except Exception: pass
        if self.cap and self.cap.isOpened():
            self.cap.release()
