import os
import cv2 as cv
import numpy as np
import pandas as pd
from core.feature_selection import select_features_mi

# TensorFlow DIHAPUS SEMENTARA AGAR TIDAK ERROR DI RASPI

class HybridDetector:
    def __init__(self, model_path=""):
        self.model = None
        self.use_tf = False
        print("⚠️ Berjalan TANPA TensorFlow (Menggunakan Mode Rule-Based).")

    def predict_cells(self, df_features, extracted_cells_imgs):
        df = df_features.copy()
        
        # Rule-Based Kuantil Fallback (Tanpa Keras/TensorFlow)
        # Menentukan sel IDA berdasarkan ukuran (Area) dan pucatan (CP_Ratio)
        area_threshold = df["Area"].quantile(0.33)
        cp_ratio_threshold = df["CP_Ratio"].quantile(0.67)
        df["IDA_Label"] = ((df["Area"] < area_threshold) & (df["CP_Ratio"] > cp_ratio_threshold)).astype(int)
            
        return df

    def run_detection_pipeline(self, df_features, extracted_cells_imgs, bboxes, raw_image, result_dir, patient_name):
        # 1. Prediksi (Pakai Rule-Based sementara)
        df_predicted = self.predict_cells(df_features, extracted_cells_imgs)
        
        # 2. Hitung statistik
        ida_count = int(df_predicted["IDA_Label"].sum())
        normal_count = len(df_predicted) - ida_count
        
        # 3. Jalankan Feature Selection
        selected_feats, mi_results, top5 = select_features_mi(df_predicted, result_dir, patient_name, target_col="IDA_Label")
        
        # 4. Visualisasi Kotak Merah / Hijau
        result_img = raw_image.copy()
        for idx, bbox in enumerate(bboxes):
            x, y, w, h = bbox
            # Cari baris yang sesuai dengan Cell_Label
            row_data = df_predicted[df_predicted["Cell_Label"] == idx + 1]
            if row_data.empty: 
                color = (128, 128, 128)
            else:
                label = row_data["IDA_Label"].values[0]
                color = (0, 0, 255) if label == 1 else (0, 255, 0)
                
            cv.rectangle(result_img, (x, y), (x + w, y + h), color, 5)
            
        detect_result_path = os.path.join(result_dir, f"detect_result_{patient_name}.png")
        cv.imwrite(detect_result_path, cv.cvtColor(result_img, cv.COLOR_RGB2BGR) if result_img.shape[-1] == 3 else result_img)
        
        return detect_result_path, ida_count, normal_count, top5
