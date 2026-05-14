import os
import shutil
import time

import cv2 as cv
import numpy as np

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QFileDialog, QPushButton,
    QRadioButton, QStackedWidget, QVBoxLayout, QCheckBox, QSpinBox, QLineEdit,
)
from PyQt5.QtGui import QPixmap
from PyQt5 import uic

from core.session  import new_session, Session
from core.pipeline import (
    run_segmentation,
    run_cell_extraction,
    run_overlap_separation,
    run_save_features,
    run_detection,
)
from core.report   import generate_pdf_report
from core.hardware import (
    SERIAL_AVAILABLE, PICAM_AVAILABLE,
    send_motor_command, close_serial,
    QPicamera2, Picamera2, open_webcam,
)
from ui.qt_helpers import (
    set_label_pixmap, display_image_on_label, display_ndarray_on_label,
)

try:
    from sensor import MagnificationSensor
    _SENSOR_OK = True
except ImportError:
    _SENSOR_OK = False

try:
    import resources_rc  # noqa: F401 – Qt resource file
except ImportError:
    pass


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("Main_Program.ui", self)

        self._init_state()
        self._init_directories()
        self._init_camera()
        self._init_sensor()
        self._bind_widgets()
        self._bind_signals()
        self._apply_styles()

        if self._using_picam and self._picam2:
            try:
                self._picam2.start()
            except Exception:
                pass

    def _init_state(self):
        """Reset all pipeline state variables."""
        self.imagePath  = None
        self.session: Session | None = None

        self._seg  = None   # SegmentationResult
        self._ext  = None   # ExtractionResult
        self._sep  = None   # SeparationResult
        self._feat = None   # FeatureResult
        self._det  = None   # DetectionResult

    def _init_directories(self):
        self.base_dir        = os.path.dirname(os.path.abspath(__file__))
        self.master_data_dir = os.path.join(
            os.path.dirname(self.base_dir), "DATA_PASIEN"
        )
        os.makedirs(self.master_data_dir, exist_ok=True)

    def _init_camera(self):
        self._using_picam = PICAM_AVAILABLE
        self._picam2      = None
        self._qpicamera2  = None
        self._cam_layout  = QVBoxLayout()

        if PICAM_AVAILABLE:
            try:
                self._picam2 = Picamera2()
                self._picam2.configure(
                    self._picam2.create_preview_configuration(
                        {"size": (480, 270)}
                    )
                )
                self._qpicamera2 = QPicamera2(
                    self._picam2, width=480, height=270, keep_ar=True
                )
            except Exception:
                self._using_picam = False

        if not self._using_picam:
            self._cap   = None
            self._timer = QTimer()
            self._timer.timeout.connect(self._update_webcam_frame)

    def _init_sensor(self):
        self._sensor = MagnificationSensor() if _SENSOR_OK else None
        self._sensor_timer = QTimer()
        if self._sensor:
            self._sensor_timer.timeout.connect(self._update_sensor_display)

    def _bind_widgets(self):
        f = self.findChild  # shorthand

        # Navigation
        self.stackedWidget = f(QStackedWidget, "stackedWidget")
        self.mainPage      = f(QPushButton,    "mainBtn")
        self.segmentPage   = f(QPushButton,    "rbcBtn")
        self.detectPage    = f(QPushButton,    "malBtn")
        self.aboutPage     = f(QPushButton,    "abtBtn")
        self.close_app     = f(QPushButton,    "closeBtn")

        # Inputs
        self.nameInput   = f(QLineEdit,      "nameInput")
        self.imageSource = [f(QRadioButton,  "camInput"),
                            f(QRadioButton,  "fileInput")]
        self.getButton   = f(QPushButton,    "getBtn")
        self.inputIm     = f(QLabel,         "rawImage")
        self.distVal     = f(QLabel,         "distVal")

        # Segmentation
        self.kmeansButton  = f(QPushButton, "kmeansBtn") or f(QPushButton, "doSegBtn")
        self.clusterText   = f(QLabel,      "clustText")
        self.selectCluster = [f(QCheckBox, f"clust{i}")   for i in range(1, 7)]
        self.clusterIm     = [f(QLabel,    f"clust{i}Im") for i in range(1, 7)]

        # Extraction
        self.extractButton = f(QPushButton, "extBtn")
        self.extractedIm   = f(QLabel,      "cellsExtract")
        self.rbcValText    = f(QLabel,      "rbcText")
        self.sepOverlap    = f(QPushButton, "overlapBtn")
        self.saveCells     = f(QPushButton, "saveBtn")

        # Detection
        self.detectButton = f(QPushButton, "detectBtn")
        self.detectText   = f(QLabel,      "detectText")
        self.detectIm     = f(QLabel,      "detectIm")
        self.visualIm     = [f(QLabel, f"vizImage_{i}") for i in range(1, 9)]
        self.pdfGenButton = f(QPushButton, "pdfBtn")

        # Motor
        self.spinBox  = f(QSpinBox,   "spinBox")
        self.upBtn    = f(QPushButton, "upBtn")
        self.downBtn  = f(QPushButton, "downBtn")
        self.stopBtn  = f(QPushButton, "stopBtn")

        if self.spinBox:
            self.spinBox.setRange(1, 99_999)
            self.spinBox.setValue(100)

    def _bind_signals(self):
        def connect(widget, signal, slot):
            if widget:
                getattr(widget, signal).connect(slot)

        # Navigation
        connect(self.mainPage,    "clicked", self._go_main)
        connect(self.segmentPage, "clicked", self._go_segment)
        connect(self.detectPage,  "clicked", self._go_detect)
        connect(self.aboutPage,   "clicked", self._go_about)
        connect(self.close_app,   "clicked", self.close)

        # Image input
        connect(self.imageSource[0], "toggled", self._on_camera_toggled)
        connect(self.imageSource[1], "toggled", self._on_file_toggled)
        connect(self.getButton,      "clicked", self.takeImage)

        # Pipeline
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

    # ── Sensor ────────────────────────────────────────────────────────────────

    def _update_sensor_display(self):
        if not self._sensor or not self.distVal:
            return
        distance = self._sensor.read_distance()
        if not np.isnan(distance):
            self.distVal.setText(f"Lens to Object Dist : {distance:.1f} mm")
        else:
            self.distVal.setText("Lens to Object Dist : Error/Out of Range")

    # Motor
    def move_up(self):
        if self.spinBox:
            send_motor_command("U", self.spinBox.value())

    def move_down(self):
        if self.spinBox:
            send_motor_command("D", self.spinBox.value())

    def stop_motor(self):
        send_motor_command("S", 0)

    # Camera
    def _on_camera_toggled(self, checked: bool):
        if not checked:
            return
        if self._sensor:
            self._sensor_timer.start(500)

        if self._using_picam and self._qpicamera2:
            if not self._qpicamera2.parent():
                self._cam_layout.setContentsMargins(0, 0, 0, 0)
                self._cam_layout.addWidget(self._qpicamera2)
                self.inputIm.setLayout(self._cam_layout)
            self.inputIm.clear()
        else:
            if self._cap is None or not self._cap.isOpened():
                self._cap = open_webcam(0)
            self._timer.start(30)
            self.inputIm.clear()

    def _on_file_toggled(self, checked: bool):
        if not checked:
            return
        self._sensor_timer.stop()
        if self.distVal:
            self.distVal.setText("Camera is not active")

        if self._using_picam and self._qpicamera2 and self._qpicamera2.parent():
            self._cam_layout.removeWidget(self._qpicamera2)
            self._qpicamera2.setParent(None)
        else:
            if self._timer.isActive():
                self._timer.stop()
            if self._cap and self._cap.isOpened():
                self._cap.release()

        self.inputIm.clear()

    def _update_webcam_frame(self):
        if not self._cap:
            return
        ret, frame = self._cap.read()
        if not ret:
            return
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        display_ndarray_on_label(self.inputIm, rgb)

    # Image acquisition
    def takeImage(self):
        patient_name = self.nameInput.text() if self.nameInput else ""
        self.session = new_session(patient_name, self.master_data_dir)
        self.imagePath = None
        save_path = self.session.raw_image_path()

        if self.imageSource[0].isChecked():          # Camera
            if self._using_picam and self._picam2:
                cfg = self._picam2.create_still_configuration(
                    main={"size": (480, 270)}
                )
                self.imagePath = save_path
                self._picam2.switch_mode_and_capture_file(
                    cfg, save_path, signal_function=self._on_capture_done
                )
            else:
                if not self._cap or not self._cap.isOpened():
                    self._cap = open_webcam(0)
                ret, frame = self._cap.read()
                if ret:
                    self.imagePath = save_path
                    cv.imwrite(save_path, frame)
                    display_image_on_label(self.inputIm, save_path)
                else:
                    self.inputIm.setText("Failed to capture image from webcam.")

        elif self.imageSource[1].isChecked():         # File picker
            dialog = QFileDialog()
            dialog.setFileMode(QFileDialog.ExistingFile)
            dialog.setNameFilter("Images (*.png *.jpg *.jpeg)")
            sample_dir = os.path.join(
                os.path.dirname(self.base_dir), "sample_raw"
            )
            if os.path.exists(sample_dir):
                dialog.setDirectory(sample_dir)

            if dialog.exec_():
                src = dialog.selectedFiles()[0]
                self.imagePath = save_path
                shutil.copy(src, save_path)
                display_image_on_label(self.inputIm, save_path)

    def _on_capture_done(self, _picam2):
        time.sleep(0.5)
        if self.imagePath and os.path.exists(self.imagePath):
            display_image_on_label(self.inputIm, self.imagePath)
            if self._qpicamera2 and self._qpicamera2.parent():
                self._cam_layout.removeWidget(self._qpicamera2)
                self._qpicamera2.setParent(None)
        else:
            print(f"❌ Capture failed: file not found at {self.imagePath}")

    # Pipeline steps
    def kmeansProcess(self):
        if not self.imagePath or not self.session:
            self.clusterText.setText("Please get an image first!")
            return

        self._go_segment()
        self.clusterText.setText("Please wait – running K-Means clustering…")
        QApplication.processEvents()

        self._seg = run_segmentation(self.imagePath)

        for idx, seg_img in enumerate(self._seg.segmented_images):
            path = self.session.cluster_image_path(idx)
            cv.imwrite(path, cv.cvtColor(seg_img, cv.COLOR_RGB2BGR))
            display_image_on_label(self.clusterIm[idx], path)

        self.clusterText.setText("K-Means clustering complete.")

    def extractCells(self):
        if self._seg is None:
            return
        self._go_extract()
        QApplication.processEvents()

        selected = [i for i, cb in enumerate(self.selectCluster)
                    if cb and cb.isChecked()]

        self._ext = run_cell_extraction(self._seg, selected)

        detect_path = self.session.detect_initial_path()
        cv.imwrite(detect_path,
                   cv.cvtColor(self._ext.annotated_image, cv.COLOR_RGB2BGR))
        display_image_on_label(self.extractedIm, detect_path)

        self.rbcValText.setText(
            f"{len(self._ext.extracted_cells)} Red Blood Cells detected. "
            "Click 'Separate Cells' if overlapping."
        )

    def separateOverlap(self):
        if self._ext is None:
            return
        self.rbcValText.setText("Separating overlapping cells with BO-FRS + GMM…")
        QApplication.processEvents()

        self._sep = run_overlap_separation(self._ext)

        sep_path = self.session.after_sep_path()
        cv.imwrite(sep_path,
                   cv.cvtColor(self._sep.annotated_image, cv.COLOR_RGB2BGR))
        display_image_on_label(self.extractedIm, sep_path)

        self.rbcValText.setText(
            f"Separation complete! "
            f"{len(self._sep.extracted_cells)} individual cells detected."
        )

    def saveExtractedCells(self):
        if self._sep is None or self._ext is None:
            return

        # Save individual cell images
        for idx, (cell_img, _, __) in enumerate(self._sep.extracted_cells):
            path = os.path.join(self.session.sep_dir, f"cell_{idx}.png")
            cv.imwrite(path, cell_img)

        self.rbcValText.setText(
            "Saving cells and extracting features, please wait…"
        )
        QApplication.processEvents()

        self._feat = run_save_features(
            sep_result=self._sep,
            rbc_only_image=self._ext.rbc_only_image,
            excel_path=self.session.features_path(),
        )

        if not self._feat.df_features.empty:
            passed = self._feat.filter_stats.get("passed", len(self._feat.df_features))
            self.rbcValText.setText(
                f"{len(self._sep.extracted_cells)} cells saved. "
                f"{passed} quality cells. Results saved."
            )
        else:
            self.rbcValText.setText("Feature extraction returned no results.")

    def detectCells(self):
        self._go_detect()
        self.detectText.setText("Running feature selection (Mutual Information)…")
        QApplication.processEvents()

        if self._feat is None or self._feat.df_features.empty:
            self.detectText.setText(
                "No feature data found.\n"
                "Please run Extract → Separate → Save first."
            )
            return

        try:
            self._det = run_detection(
                feat_result=self._feat,
                cell_info=self._sep.cell_info,
                raw_image=self._seg.raw_image,
                selected_path=self.session.features_selected_path(),
                mi_path=self.session.mutual_info_path(),
            )
        except Exception as e:
            self.detectText.setText(f"Feature selection failed: {e}")
            return

        # Save + display annotated result
        det_path = self.session.detect_result_path()
        cv.imwrite(det_path,
                   cv.cvtColor(self._det.annotated_image, cv.COLOR_RGB2BGR))
        display_image_on_label(self.detectIm, det_path)

        for lbl in self.visualIm:
            if lbl:
                lbl.clear()
        for slot, (cell_img, _, __) in enumerate(
                self._sep.extracted_cells[:8]):
            if cell_img.ndim == 3:
                display_ndarray_on_label(
                    self.visualIm[slot],
                    cv.cvtColor(cell_img, cv.COLOR_BGR2RGB),
                )

        top5 = self._det.top_features
        self.detectText.setText(
            f"Feature Selection Complete!\n"
            f"Total: {len(self._feat.df_features)}  |  "
            f"IDA: {self._det.ida_count}  |  "
            f"Normal: {self._det.normal_count}\n"
            "Top features:\n"
            + "\n".join(f"  {i+1}. {f}" for i, f in enumerate(top5))
        )

    def generatePDF(self):
        if not self.session or self._det is None:
            return

        pdf_path = generate_pdf_report(
            base_dir=os.path.dirname(self.base_dir),
            image_path=self.imagePath,
            detect_path=self.session.detect_result_path(),
            cells=len(self._sep.cell_info) if self._sep else 0,
            mal=self._det.ida_count,
            par_path=self.session.sep_dir,
            output_path=self.session.pdf_report_path(),
            patient_name=self.session.patient_name,
        )
        self.detectText.setText(
            f"PDF report saved to:\n{self.session.res_dir}"
        )

    def _go_main(self):
        if self.stackedWidget:
            self.stackedWidget.setCurrentIndex(0)

    def _go_segment(self):
        if self.stackedWidget:
            self.stackedWidget.setCurrentIndex(1)

    def _go_extract(self):
        if self.stackedWidget:
            self.stackedWidget.setCurrentIndex(2)

    def _go_detect(self):
        if self.stackedWidget:
            self.stackedWidget.setCurrentIndex(3)

    def _go_about(self):
        if self.stackedWidget:
            self.stackedWidget.setCurrentIndex(4)

    def _apply_styles(self):
        menu_style = (
            "QPushButton {border:1px; color: rgb(225,225,225); "
            "background-color: rgb(17,70,143)}"
            "QPushButton:hover {background-color: rgb(35,56,148); "
            "color: rgb(255,255,255);}"
            "QPushButton:checked {background-color: rgb(4,21,98); "
            "color: rgb(255,255,255);}"
        )
        action_style = (
            "QPushButton {border:1px; border-radius:10px; color: rgb(0,0,0); "
            "background-color: rgb(214,222,255);}"
            "QPushButton:hover {background-color: rgb(35,56,148); "
            "color: rgb(255,255,255);}"
            "QPushButton:checked {background-color: rgb(4,21,98); "
            "color: rgb(255,255,255);}"
        )
        for btn in (self.mainPage, self.aboutPage):
            if btn:
                btn.setStyleSheet(menu_style)
        for btn in (self.segmentPage, self.detectPage):
            if btn:
                btn.setStyleSheet(action_style)

    def closeEvent(self, event):
        if self._using_picam and self._picam2:
            try:
                self._picam2.stop()
            except Exception:
                pass
        else:
            if hasattr(self, "_timer") and self._timer.isActive():
                self._timer.stop()
            if self._cap and self._cap.isOpened():
                self._cap.release()

        self._sensor_timer.stop()
        close_serial()
        super().closeEvent(event)
