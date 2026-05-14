import os
import time
from fpdf import FPDF

class PDFWithHeaderFooter(FPDF):
    def __init__(self, base_dir):
        super().__init__()
        self.base_dir = base_dir
        self.add_font("Poppins", "", os.path.join(self.base_dir, "add-on/Poppins-Bold.ttf"), uni=True)
        self.add_font("Inter", "", os.path.join(self.base_dir, "add-on/Inter_18pt-SemiBold.ttf"), uni=True)
        self.timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

    def header(self):
        self.set_font("Poppins", "", 20)
        self.set_fill_color(4, 21, 98)
        self.set_xy(0, 16)
        self.cell(165, 3, fill=True)
        
        logo_path = os.path.join(self.base_dir, "add-on/logo.png")
        if os.path.exists(logo_path):
            self.image(logo_path, x=170, y=10, w=30)
            
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

    def generate_result(self, imagePath, detectPath, cells, mal, parPath, output_path, patient_name):
        self.add_page()
        self.set_font("Inter", size=12)
        self.set_xy(18, 40)
        self.set_text_color(0)
        self.cell(170, 10, f"Patient Name / ID: {patient_name.replace('_', ' ')}", ln=True)
        
        self.set_xy(18, 50)
        self.set_text_color(120)
        self.cell(170, 10, f"Report generated on {self.timestamp}", ln=True)

        if imagePath and os.path.exists(imagePath): self.image(imagePath, x=18, y=70, w=88, h=49.5)
        if detectPath and os.path.exists(detectPath): self.image(detectPath, x=100, y=70, w=88, h=49.5)

        self.set_font_size(10)
        self.set_text_color(150)
        self.set_xy(18, 123)
        self.multi_cell(170, 5, "Green bounding boxes indicate normal red blood cells, while red bounding boxes indicate IDA/malaria-infected cells.")

        self.set_text_color(0)
        self.set_xy(18, 135)
        self.set_font_size(14)
        self.cell(170, 10, f"Total red blood cells detected: {cells}", border=1, ln=True)
        self.set_x(18)
        self.cell(170, 10, f"Infected/Abnormal cells detected: {mal}", border=1, ln=True)

        if parPath and os.path.exists(parPath) and os.path.isdir(parPath):
            image_files = os.listdir(parPath)[:8]
            for index, filename in enumerate(image_files):
                col = index % 4
                row = index // 4
                x = 40 + col * 32
                y = 163 + row * 32
                file_path = os.path.join(parPath, filename)
                self.image(file_path, x=x, y=y, w=30, h=30)

        self.set_text_color(255)
        if mal != 0:
            self.set_xy(18, 230)
            self.set_fill_color(255, 0, 0)
            self.multi_cell(170, 6, "Based on our system's detection results, the patient is identified as having abnormalities (IDA) and requires further clinical evaluation.", border=1, fill=True)
        else:
            self.set_xy(18, 170)
            self.set_fill_color(0, 255, 0)
            self.multi_cell(170, 6, "Our system's detection results indicate normal cells in the patient.", border=1, fill=True)

        self.output(output_path)
