import sys
import os
import time
import shutil
import numpy as np
import cv2 as cv
import pandas as pd
import imageio
from PIL import Image
from sklearn.feature_selection import mutual_info_classif
from fpdf import FPDF

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QPushButton,
    QRadioButton, QStackedWidget, QVBoxLayout, QCheckBox, QSpinBox, QLineEdit
)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic

from segmentyanes import *
from sensor import *
from feature_extraction import run_feature_extraction
import resources_rc

# Communication
import serial
SERIAL_AVAILABLE = False
esp_serial = None

for port in ("/dev/ttyUSB0", "/dev/ttyACM0"):
    try:
        esp_serial = serial.Serial(port, 115200, timeout=1)
        SERIAL_AVAILABLE = True
        print(f"✅ ESP32 connected via {port}")
        time.sleep(2)
        break
    except Exception:
        continue

if not SERIAL_AVAILABLE:
    print("❌ ESP32 not detected. Running in Motor Simulation Mode.")

#Camera
try:
    from picamera2.previews.qt import QPicamera2
    from picamera2 import Picamera2
    PICAM_AVAILABLE = True
except Exception:
    QPicamera2 = None
    Picamera2 = None
    PICAM_AVAILABLE = False


#Report
class PDFWithHeaderFooter(FPDF):
    """Generates a formatted PDF report with header, footer, images, and results."""

    def __init__(self, base_dir: str):
        super().__init__()
        self.base_dir = base_dir
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        addon = os.path.join(self.base_dir, "add-on")
        self.add_font("Poppins", "", os.path.join(addon, "Poppins-Bold.ttf"), uni=True)
        self.add_font("Inter",   "", os.path.join(addon, "Inter_18pt-SemiBold.ttf"), uni=True)

    #FPDF
    def header(self):
        self.set_font("Poppins", "", 20)
        self.set_fill_color(4, 21, 98)
        self.set_xy(0, 16)
        self.cell(165, 3, fill=True)

        logo = os.path.join(self.base_dir, "add-on/logo.png")
        if os.path.exists(logo):
            self.image(logo, x=170, y=10, w=30)

        self.ln(10)
        self.cell(0, 10, "Segmentation and Detection Report", ln=True, align="L")
        self.ln(5)

    def footer(self):
        self.set_y(-20)
        self.set_font("Poppins", "", 15)
        self.set_text_color(100)
        self.cell(0, 10, "MalaScope, 2026", align="L")
        self.set_xy(65, 281)
        self.set_fill_color(4, 21, 98)
        self.cell(170, 2, fill=True)

    # Result
    def generate_result(
        self,
        image_path: str,
        detect_path: str,
        cells: int,
        mal: int,
        par_path: str,
        output_path: str,
        patient_name: str,
    ):
        self.add_page()
        self.set_font("Inter", size=12)
        self.set_text_color(0)

        # Patient info
        self.set_xy(18, 40)
        self.cell(170, 10, f"Patient Name / ID: {patient_name.replace('_', ' ')}", ln=True)
        self.set_xy(18, 50)
        self.set_text_color(120)
        self.cell(170, 10, f"Report generated on {self.timestamp}", ln=True)

        # Images
        if os.path.exists(image_path):
            self.image(image_path,  x=18,  y=70, w=88, h=49.5)
        if os.path.exists(detect_path):
            self.image(detect_path, x=100, y=70, w=88, h=49.5)

        # Caption
        self.set_font_size(10)
        self.set_text_color(150)
        self.set_xy(18, 123)
        self.multi_cell(
            170, 5,
            "Green bounding boxes = normal RBCs; red bounding boxes = IDA/infected cells."
        )

        # Summary cells
        self.set_text_color(0)
        self.set_xy(18, 135)
        self.set_font_size(14)
        self.cell(170, 10, f"Total red blood cells detected: {cells}",      border=1, ln=True)
        self.set_x(18)
        self.cell(170, 10, f"Infected/Abnormal cells detected: {mal}",       border=1, ln=True)

        # Cropped cell thumbnails (up to 8)
        if os.path.isdir(par_path):
            for idx, filename in enumerate(os.listdir(par_path)[:8]):
                col, row = idx % 4, idx // 4
                self.image(
                    os.path.join(par_path, filename),
                    x=40 + col * 32,
                    y=163 + row * 32,
                    w=30, h=30,
                )

        # Conclusion banner
        self.set_text_color(255)
        if mal != 0:
            self.set_xy(18, 230)
            self.set_fill_color(255, 0, 0)
            self.multi_cell(
                170, 6,
                "Based on our system's detection results, the patient is identified as having "
                "abnormalities (IDA) and requires further clinical evaluation.",
                border=1, fill=True,
            )
        else:
            self.set_xy(18, 170)
            self.set_fill_color(0, 255, 0)
            self.multi_cell(
                170, 6,
                "Our system's detection results indicate normal cells in the patient.",
                border=1, fill=True,
            )

        self.output(output_path)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("Main_Program.ui", self)

        self._init_directories()
        self._init_camera()
        self._bind_widgets()
        self._bind_signals()
        self.setStyles()

        if self.using_picam and self.picam2:
            try:
                self.picam2.start()
            except Exception:
                pass

        self.sensor = MagnificationSensor()
        self.sensor_timer = QTimer()
        self.sensor_timer.timeout.connect(self._update_sensor_value)

    def _init_directories(self):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.master_data_dir = os.path.join(self.base_dir, "DATA_PASIEN")
        os.makedirs(self.master_data_dir, exist_ok=True)
      
        self.current_raw_dir   = None
        self.current_clust_dir = None
        self.current_sep_dir   = None
        self.current_res_dir   = None
        self.current_patient   = "Anonim"

    def _init_camera(self):
        self.using_picam = PICAM_AVAILABLE
        self.picam2      = None
        self.qpicamera2  = None

        if PICAM_AVAILABLE:
            try:
                self.picam2 = Picamera2()
                self.picam2.configure(
                    self.picam2.create_preview_configuration({"size": (480, 270)})
                )
                self.qpicamera2 = QPicamera2(self.picam2, width=480, height=270, keep_ar=True)
            except Exception:
                self.using_picam = False

        if not self.using_picam:
            self.cap = cv.VideoCapture(0)
            self.timer = QTimer()
            self.timer.timeout.connect(self._update_webcam_frame)

    def _bind_widgets(self):
        """Cache all Qt widget references from the .ui file."""
        f = self.findChild  # shorthand

        # Navigation
        self.stackedWidget = f(QStackedWidget, "stackedWidget")
        self.mainPage      = f(QPushButton, "mainBtn")
        self.segmentPage   = f(QPushButton, "rbcBtn")
        self.detectPage    = f(QPushButton, "malBtn")
        self.aboutPage     = f(QPushButton, "abtBtn")
        self.close_app     = f(QPushButton, "closeBtn")

        # Inputs
        self.nameInput     = f(QLineEdit,   "nameInput")
        self.imageSource   = [f(QRadioButton, "camInput"), f(QRadioButton, "fileInput")]
        self.getButton     = f(QPushButton, "getBtn")
        self.inputIm       = f(QLabel,      "rawImage")
        self.distVal       = f(QLabel,      "distVal")

        # Segmentation
        self.kmeansButton  = f(QPushButton, "kmeansBtn") or f(QPushButton, "doSegBtn")
        self.clusterText   = f(QLabel,      "clustText")
        self.selectCluster = [f(QCheckBox, f"clust{i}")  for i in range(1, 7)]
        self.clusterIm     = [f(QLabel,    f"clust{i}Im") for i in range(1, 7)]
        self.layout        = QVBoxLayout()

        # Extraction
        self.extractButton = f(QPushButton, "extBtn")
        self.extractedIm   = f(QLabel,      "cellsExtract")
        self.rbcValText    = f(QLabel,      "rbcText")
        self.sepOverlap    = f(QPushButton, "overlapBtn")
        self.saveCells     = f(QPushButton, "saveBtn")

        # Detection
        self.detectButton  = f(QPushButton, "detectBtn")
        self.detectText    = f(QLabel,      "detectText")
        self.detectIm      = f(QLabel,      "detectIm")
        self.visualIm      = [f(QLabel, f"vizImage_{i}") for i in range(1, 9)]
        self.pdfGenButton  = f(QPushButton, "pdfBtn")

        # Motor control
        self.spinBox  = f(QSpinBox,   "spinBox")
        self.upBtn    = f(QPushButton, "upBtn")
        self.downBtn  = f(QPushButton, "downBtn")
        self.stopBtn  = f(QPushButton, "stopBtn")

        if self.spinBox:
            self.spinBox.setRange(1, 99999)
            self.spinBox.setValue(100)

    def _bind_signals(self):
        def connect(widget, signal, slot):
            if widget:
                getattr(widget, signal).connect(slot)

        # Navigation
        connect(self.mainPage,    "clicked", self.moveMainPage)
        connect(self.segmentPage, "clicked", self.moveSegmentPage)
        connect(self.detectPage,  "clicked", self.moveDetectPage)
        connect(self.aboutPage,   "clicked", self.moveAboutPage)
        connect(self.close_app,   "clicked", self.closeApp)

        # Image input
        connect(self.imageSource[0], "toggled", self._on_camera_toggled)
        connect(self.imageSource[1], "toggled", self._on_file_toggled)
        connect(self.getButton,      "clicked", self.takeImage)

        # Processing pipeline
        connect(self.kmeansButton,  "clicked", self.kmeansProcess)
        connect(self.extractButton, "clicked", self.extractCells)
        connect(self.sepOverlap,    "clicked", self.separateOverlap)
        connect(self.saveCells,     "clicked", self.saveExtractedCells)
        connect(self.detectButton,  "clicked", self.detectCells)
        connect(self.pdfGenButton,  "clicked", self.generatePDF)

        # Motor
        connect(self.upBtn,   "clicked", self.move_up)
        connect(self.downBtn, "clicked", self.move_down)
        connect(self.stopBtn, "clicked", self.stop_motor)

    def _create_session_folders(self):
        name = (self.nameInput.text().strip() if self.nameInput else "").replace(" ", "_")
        self.current_patient = name or "Anonim"

        session = os.path.join(
            self.master_data_dir,
            f"{self.current_patient}_{time.strftime('%Y%m%d_%H%M%S')}",
        )
        self.current_raw_dir   = os.path.join(session, "0_raw_image")
        self.current_clust_dir = os.path.join(session, "1_clustering_image")
        self.current_sep_dir   = os.path.join(session, "2_separated_cells")
        self.current_res_dir   = os.path.join(session, "3_results")

        for folder in (
            self.current_raw_dir, self.current_clust_dir,
            self.current_sep_dir, self.current_res_dir,
        ):
            os.makedirs(folder, exist_ok=True)

    #Sensor Control
    def _update_sensor_value(self):
        distance = self.sensor.read_distance()
        if self.distVal:
            if not np.isnan(distance):
                self.distVal.setText(f"Lens to Object Dist : {distance:.1f} mm")
            else:
                self.distVal.setText("Lens to Object Dist : Error/Out of Range")

    # Motor Stepper Control
    def _send_esp_command(self, direction: str, steps: int):
        cmd = f"{direction}{steps}\n"
        if SERIAL_AVAILABLE and esp_serial and esp_serial.is_open:
            try:
                esp_serial.write(cmd.encode("utf-8"))
                print(f"📡 → ESP32: {cmd.strip()}")
            except Exception as e:
                print(f"❌ Serial send failed: {e}")
        else:
            print(f"⚠️  Simulation – Motor {direction} {steps} steps")

    def move_up(self):
        if self.spinBox:
            self._send_esp_command("U", self.spinBox.value())

    def move_down(self):
        if self.spinBox:
            self._send_esp_command("D", self.spinBox.value())

    def stop_motor(self):
        print("🛑 Stop pressed")
        if SERIAL_AVAILABLE and esp_serial and esp_serial.is_open:
            esp_serial.write(b"S0\n")

    # Camera Raspi Control
    def _on_camera_toggled(self, checked: bool):
        if not checked:
            return
        self.sensor_timer.start(500)

        if self.using_picam and self.qpicamera2:
            if not self.qpicamera2.parent():
                self.layout.setContentsMargins(0, 0, 0, 0)
                self.layout.addWidget(self.qpicamera2)
                self.inputIm.setLayout(self.layout)
            self.inputIm.clear()
        else:
            if not hasattr(self, "cap") or not self.cap.isOpened():
                self.cap = cv.VideoCapture(0)
            self.timer.start(30)
            self.inputIm.clear()

    def _on_file_toggled(self, checked: bool):
        if not checked:
            return
        self.sensor_timer.stop()
        if self.distVal:
            self.distVal.setText("Camera is not active")

        if self.using_picam and self.qpicamera2 and self.qpicamera2.parent():
            self.layout.removeWidget(self.qpicamera2)
            self.qpicamera2.setParent(None)
        else:
            if hasattr(self, "timer") and self.timer.isActive():
                self.timer.stop()
            if hasattr(self, "cap") and self.cap.isOpened():
                self.cap.release()

        self.inputIm.clear()

    def takeImage(self):
        self._create_session_folders()
        self.imagePath = None
        save_name = f"raw_{self.current_patient}.jpg"

        if self.imageSource[0].isChecked():  # Camera
            if self.using_picam and self.picam2:
                cfg = self.picam2.create_still_configuration(main={"size": (480, 270)})
                self.imagePath = os.path.join(self.current_raw_dir, save_name)
                self.picam2.switch_mode_and_capture_file(
                    cfg, self.imagePath, signal_function=self._on_capture_done
                )
            else:
                if not hasattr(self, "cap") or not self.cap.isOpened():
                    self.cap = cv.VideoCapture(0)
                ret, frame = self.cap.read()
                if ret:
                    self.imagePath = os.path.join(self.current_raw_dir, save_name)
                    cv.imwrite(self.imagePath, frame)
                    self._display_image(self.imagePath)
                else:
                    self.inputIm.setText("Failed to capture image from webcam.")

        elif self.imageSource[1].isChecked():  # File
            dialog = QFileDialog()
            dialog.setFileMode(QFileDialog.ExistingFile)
            dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
            sample_dir = os.path.join(self.base_dir, "sample_raw")
            if os.path.exists(sample_dir):
                dialog.setDirectory(sample_dir)

            if dialog.exec_():
                src = dialog.selectedFiles()[0]
                self.imagePath = os.path.join(self.current_raw_dir, save_name)
                shutil.copy(src, self.imagePath)
                self._display_image(self.imagePath)

    def _on_capture_done(self, picam2):
        self.imagePath = os.path.join(self.current_raw_dir, f"raw_{self.current_patient}.jpg")
        time.sleep(0.5)
        if os.path.exists(self.imagePath):
            Image.open(self.imagePath).save(self.imagePath)
            self._display_image(self.imagePath)
            if self.qpicamera2.parent():
                self.layout.removeWidget(self.qpicamera2)
                self.qpicamera2.setParent(None)
        else:
            print(f"❌ Capture failed: file not found at {self.imagePath}")

    def _update_webcam_frame(self):
        if not hasattr(self, "cap"):
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qt_img = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        self._set_pixmap(self.inputIm, QPixmap.fromImage(qt_img))

    def _display_image(self, path: str):
        self._set_pixmap(self.inputIm, QPixmap(path))

    def _set_pixmap(self, label: QLabel, pixmap: QPixmap):
        label.setPixmap(
            pixmap.scaled(label.width(), label.height(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )
        label.setAlignment(Qt.AlignCenter)

    # Pipeline
    def kmeansProcess(self):
        if not self.current_clust_dir:
            self.clusterText.setText("Please get an image first!")
            return

        self.moveSegmentPage()
        self.segmentPage.setChecked(True)
        self.clusterText.setText("Please wait – running k-means clustering…")
        QApplication.processEvents()

        self.raw_image = cv.cvtColor(imageio.imread(self.imagePath), cv.COLOR_BGR2RGB)
        self.hsv_clean_image, _ = convert_hsv_circular(self.raw_image, v_thresh=20)
        self.segmented_images, self.labels_full = kmeans_segmentation(
            self.hsv_clean_image, k=6, use_preprocessing=True, v_thresh=20
        )

        for idx, seg_img in enumerate(self.segmented_images):
            path = os.path.join(self.current_clust_dir, f"cluster_{idx + 1}.jpg")
            cv.imwrite(path, cv.cvtColor(seg_img, cv.COLOR_RGB2BGR))
            self._set_pixmap(self.clusterIm[idx], QPixmap(path))

        self.clusterText.setText("K-means clustering complete.")

    def extractCells(self):
        self.moveExtractPage()
        QApplication.processEvents()

        selected = [i for i, cb in enumerate(self.selectCluster) if cb.isChecked()]
        rgb_clean = cv.cvtColor(self.hsv_clean_image, cv.COLOR_HSV2RGB)

        self.rbc_only_image, self.filtered_mask, _ = remove_unwanted_cells_extended(
            self.segmented_images, selected, rgb_clean
        )

        gray = cv.cvtColor(self.rbc_only_image, cv.COLOR_RGB2GRAY)
        edge_map, contour_edge = sobel_edge_detect(gray)
        cells_detected = draw_bounding_boxes(self.rbc_only_image, contour_edge)

        for idx, contour in enumerate(extract_contours(gray, edge_map)[0], start=1):
            x, y, w, h = cv.boundingRect(contour)
            cx, cy = x + w // 2, y + h // 2
            font, scale, thick = cv.FONT_HERSHEY_SIMPLEX, 0.6, 2
            (lw, lh), _ = cv.getTextSize(str(idx), font, scale, thick)
            pad = 4
            lx = max(0, min(cx - lw // 2 - pad, cells_detected.shape[1] - lw - 2 * pad))
            ly = max(lh + 2 * pad, min(cy - lh // 2 - pad, cells_detected.shape[0]))
            cv.rectangle(cells_detected, (lx, ly - lh - pad), (lx + lw + 2 * pad, ly + pad), (0, 0, 0), -1)
            cv.putText(cells_detected, str(idx), (lx + pad, ly - pad), font, scale, (255, 255, 255), thick, cv.LINE_AA)

        detect_path = os.path.join(self.current_res_dir, "detect_cells_initial.jpg")
        cv.imwrite(detect_path, cells_detected)
        self._set_pixmap(self.extractedIm, QPixmap(detect_path))

        contours, _ = extract_contours(gray, edge_map)
        self.extracted_cells    = []
        self.cell_masks_list    = []
        self.bounding_boxes_sep = []

        for contour in contours:
            mask = np.zeros(self.rbc_only_image.shape[:2], dtype=np.uint8)
            cv.drawContours(mask, [contour], -1, 255, -1)
            x, y, w, h = cv.boundingRect(contour)
            cell = cv.bitwise_and(
                self.rbc_only_image[y:y+h, x:x+w],
                self.rbc_only_image[y:y+h, x:x+w],
                mask=mask[y:y+h, x:x+w],
            )
            self.extracted_cells.append((cell, x, y))
            self.cell_masks_list.append(mask[y:y+h, x:x+w])
            self.bounding_boxes_sep.append((x, y, w, h))

        self.rbcValText.setText(
            f"{len(self.extracted_cells)} Red Blood Cells detected. "
            "Click 'Separate Cells' if overlapping."
        )

    def separateOverlap(self):
        self.rbcValText.setText("Separating overlapping cells with BO-FRS + GMM…")
        QApplication.processEvents()

        opened_mask  = bounded_opening(self.filtered_mask, num_openings=3)
        bofrs_result = bounded_opening_frs(opened_mask, num_openings=3)
        cropped_cells, bboxes, cell_masks = separate_overlapping_rbc_with_gmm(
            bofrs_result, self.rbc_only_image
        )

        self.extracted_cells    = []
        self.cell_info          = []
        self.cell_masks_list    = []
        self.bounding_boxes_sep = []

        for idx, (cell_img, bbox) in enumerate(zip(cropped_cells, bboxes)):
            x, y, w, h = bbox
            self.extracted_cells.append((cell_img, x, y))
            self.cell_info.append({"filename": f"cell_{idx}.png", "bbox": [x, y, w, h]})
            self.cell_masks_list.append(cell_masks[idx])
            self.bounding_boxes_sep.append(bbox)
          
        annotated = self.rbc_only_image.copy()
        for idx, (x, y, w, h) in enumerate(bboxes, start=1):
            cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 5)
            font, scale, thick = cv.FONT_HERSHEY_SIMPLEX, 0.6, 2
            (lw, lh), _ = cv.getTextSize(str(idx), font, scale, thick)
            pad = 4
            cx, cy = x + w // 2, y + h // 2
            lx = max(0, min(cx - lw // 2 - pad, annotated.shape[1] - lw - 2 * pad))
            ly = max(lh + 2 * pad, min(cy - lh // 2 - pad, annotated.shape[0]))
            cv.rectangle(annotated, (lx, ly - lh - pad), (lx + lw + 2 * pad, ly + pad), (0, 0, 0), -1)
            cv.putText(annotated, str(idx), (lx + pad, ly - pad), font, scale, (255, 255, 255), thick, cv.LINE_AA)

        sep_path = os.path.join(self.current_res_dir, "after_sep.jpg")
        cv.imwrite(sep_path, annotated)
        self._set_pixmap(self.extractedIm, QPixmap(sep_path))
        self.rbcValText.setText(f"Separation complete! {len(self.extracted_cells)} individual cells detected.")

    def saveExtractedCells(self):
        self.cell_info = []
        for idx, (cell_img, x, y) in enumerate(self.extracted_cells):
            h, w = cell_img.shape[:2]
            path = os.path.join(self.current_sep_dir, f"cell_{idx}.png")
            cv.imwrite(path, cell_img)
            self.cell_info.append({"filename": f"cell_{idx}.png", "bbox": [x, y, w, h]})

        self.rbcValText.setText("Saving cells and extracting features, please wait…")
        QApplication.processEvents()

        excel_path = os.path.join(self.current_res_dir, f"features_{self.current_patient}.xlsx")
        try:
            df, _, filter_stats = run_feature_extraction(
                extracted_cells=self.extracted_cells,
                bounding_boxes=self.bounding_boxes_sep,
                cell_masks=self.cell_masks_list,
                img_shape=self.rbc_only_image.shape,
                output_csv_path=None,
            )
            if not df.empty:
                df.to_excel(excel_path, index=False)
                self.df_features = df
                passed = filter_stats.get("passed", len(df))
                self.rbcValText.setText(
                    f"{len(self.extracted_cells)} cells saved. "
                    f"{passed} quality cells. Results saved."
                )
            else:
                self.rbcValText.setText("Feature extraction returned no results.")
        except Exception as e:
            self.rbcValText.setText(f"Feature extraction failed: {e}")

    def detectCells(self):
        self.moveDetectPage()
        self.detectPage.setChecked(True)
        self.detectText.setText("Running feature selection (Mutual Information)…")
        QApplication.processEvents()

        if not hasattr(self, "df_features") or self.df_features.empty:
            self.detectText.setText(
                "No feature data found.\nPlease run Extract → Separate → Save first."
            )
            return

        try:
            df = self.df_features.copy()

            # Rule-based IDA labelling
            area_thresh = df["Area"].quantile(0.33)
            cp_thresh   = df["CP_Ratio"].quantile(0.67)
            df["IDA_Label"] = (
                (df["Area"] < area_thresh) & (df["CP_Ratio"] > cp_thresh)
            ).astype(int)

            ida_count    = int(df["IDA_Label"].sum())
            normal_count = len(df) - ida_count

            # Mutual information feature selection
            exclude      = {"Cell_Label", "X", "Y", "IDA_Label"}
            feature_cols = [c for c in df.columns if c not in exclude]
            mi_scores    = mutual_info_classif(
                df[feature_cols].fillna(0), df["IDA_Label"], random_state=42
            )
            mi_df = (
                pd.DataFrame({"Feature": feature_cols, "MI_Score": mi_scores})
                .sort_values("MI_Score", ascending=False)
                .reset_index(drop=True)
            )
            selected = mi_df[mi_df["MI_Score"] > 0.01]["Feature"].tolist()

            # Save feature tables
            df_sel = df[["Cell_Label", "X", "Y"] + selected + ["IDA_Label"]]
            df_sel.to_excel(
                os.path.join(self.current_res_dir, f"features_selected_{self.current_patient}.xlsx"),
                index=False,
            )
            mi_df.to_excel(
                os.path.join(self.current_res_dir, f"mutual_info_{self.current_patient}.xlsx"),
                index=False,
            )
            self.df_selected = df_sel

            # Summary text
            top5 = mi_df.head(5)["Feature"].tolist()
            self.detectText.setText(
                f"Feature Selection Complete!\n"
                f"Total: {len(df)}  |  IDA: {ida_count}  |  Normal: {normal_count}\n"
                "Top features:\n" + "\n".join(f"  {i+1}. {f}" for i, f in enumerate(top5))
            )

            # Annotate result image
            annotated = self.raw_image.copy()
            COLOR = {0: (0, 255, 0), 1: (0, 0, 255), -1: (128, 128, 128)}
            for row_idx, info in enumerate(self.cell_info):
                x, y, w, h = info["bbox"]
                row_data = df[df["Cell_Label"] == row_idx + 1]
                label = int(row_data["IDA_Label"].values[0]) if not row_data.empty else -1
                cv.rectangle(annotated, (x, y), (x + w, y + h), COLOR[label], 5)

            self.detectResultPath = os.path.join(
                self.current_res_dir, f"detect_result_{self.current_patient}.png"
            )
            cv.imwrite(self.detectResultPath, annotated)
            self._set_pixmap(self.detectIm, QPixmap(self.detectResultPath))

            # Thumbnail strip
            for lbl in self.visualIm:
                lbl.clear()
            for slot, (cell_img, _, __) in enumerate(self.extracted_cells[:8]):
                h_img, w_img = cell_img.shape[:2]
                if h_img == 0 or w_img == 0:
                    continue
                rgb = cv.cvtColor(cell_img, cv.COLOR_BGR2RGB) if cell_img.ndim == 3 else cell_img
                qt_img = QImage(rgb.data, w_img, h_img, rgb.strides[0], QImage.Format_RGB888)
                self._set_pixmap(self.visualIm[slot], QPixmap.fromImage(qt_img))

        except Exception as e:
            self.detectText.setText(f"Feature selection failed: {e}")

    def generatePDF(self):
        pdf_path = os.path.join(self.current_res_dir, f"Report_{self.current_patient}.pdf")

        PDFWithHeaderFooter(self.base_dir).generate_result(
            image_path=self.imagePath,
            detect_path=self.detectResultPath,
            cells=len(self.cell_info) if hasattr(self, "cell_info") else 0,
            mal=len(self.df_features) if hasattr(self, "df_features") else 0,
            par_path=self.current_sep_dir,
            output_path=pdf_path,
            patient_name=self.current_patient,
        )
        self.detectText.setText(f"PDF report saved to:\n{self.current_res_dir}")

    def moveMainPage(self):    self.stackedWidget and self.stackedWidget.setCurrentIndex(0)
    def moveSegmentPage(self): self.stackedWidget and self.stackedWidget.setCurrentIndex(1)
    def moveExtractPage(self): self.stackedWidget and self.stackedWidget.setCurrentIndex(2)
    def moveDetectPage(self):  self.stackedWidget and self.stackedWidget.setCurrentIndex(3)
    def moveAboutPage(self):   self.stackedWidget and self.stackedWidget.setCurrentIndex(4)

    # Addition
    def setStyles(self):
        menu_style = (
            "QPushButton {border:1px; color: rgb(225,225,225); background-color: rgb(17,70,143)}"
            "QPushButton:hover {background-color: rgb(35,56,148); color: rgb(255,255,255);}"
            "QPushButton:checked {background-color: rgb(4,21,98); color: rgb(255,255,255);}"
        )
        action_style = (
            "QPushButton {border:1px; border-radius:10px; color: rgb(0,0,0); background-color: rgb(214,222,255);}"
            "QPushButton:hover {background-color: rgb(35,56,148); color: rgb(255,255,255);}"
            "QPushButton:checked {background-color: rgb(4,21,98); color: rgb(255,255,255);}"
        )
        for btn in (self.mainPage, self.aboutPage):
            if btn: btn.setStyleSheet(menu_style)
        for btn in (self.segmentPage, self.detectPage):
            if btn: btn.setStyleSheet(action_style)

    # Reset
    def closeApp(self):
        self.close()

    def closeEvent(self, event):
        if self.using_picam and self.picam2:
            try:
                self.picam2.stop()
            except Exception:
                pass
        else:
            if hasattr(self, "timer") and self.timer.isActive():
                self.timer.stop()
            if hasattr(self, "sensor_timer") and self.sensor_timer.isActive():
                self.sensor_timer.stop()
            if hasattr(self, "cap") and self.cap.isOpened():
                self.cap.release()

        if esp_serial and esp_serial.is_open:
            esp_serial.close()

        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
