import os
import cv2 as cv
import numpy as np
import pandas as pd
from core.feature_selection import select_features_mi

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf

class HybridDetector:
    def __init__(self, model_path="/home/microscope/malaria/MalaScope/fold_5.keras"):
        self.model = None
        self.use_tf = False
        try:
            self.model = tf.keras.models.load_model(model_path)
            self.use_tf = True
            print("✅ TensorFlow Model berhasil di-load.")
        except Exception as e:
            print(f"⚠️ TensorFlow Model gagal di-load, menggunakan Rule-Based Fallback. Error: {e}")

    def predict_cells(self, df_features, extracted_cells_imgs):
        df = df_features.copy()
        
        if self.use_tf:
            ida_labels = []
            target_size = 128
            for idx, row in df.iterrows():
                # Dapatkan indeks asli gambar dari dataframe (1-indexed ke 0-indexed)
                cell_idx = int(row["Cell_Label"]) - 1
                if cell_idx < len(extracted_cells_imgs):
                    cell_img = extracted_cells_imgs[cell_idx]
                    img_resized = cv.resize(cell_img, (target_size, target_size))
                    img_data = np.expand_dims(img_resized.astype('float32') / 255.0, axis=0)
                    
                    preds = self.model.predict(img_data, verbose=0)
                    score = preds[0][0] # Sesuaikan dimensi output layer modelmu
                    label_int = 1 if score < 0.25 else 0 # 1 = IDA, 0 = Normal
                    ida_labels.append(label_int)
                else:
                    ida_labels.append(0)
            df["IDA_Label"] = ida_labels

        else:
            # Rule-Based Kuantil Fallback (Sesuai aslimu)
            area_threshold = df["Area"].quantile(0.33)
            cp_ratio_threshold = df["CP_Ratio"].quantile(0.67)
            df["IDA_Label"] = ((df["Area"] < area_threshold) & (df["CP_Ratio"] > cp_ratio_threshold)).astype(int)
            
        return df

    def run_detection_pipeline(self, df_features, extracted_cells_imgs, bboxes, raw_image, result_dir, patient_name):
        # 1. Prediksi (ML / Rule-Based)
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
