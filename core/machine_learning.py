import os
import cv2 as cv
import numpy as np
import joblib
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops

class SVMDetector:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        
        # Arahkan ke folder models di dalam source
        models_dir = os.path.join(self.base_dir, "source", "models")
        
        model_path = os.path.join(models_dir, "model.pkl")
        scaler_path = os.path.join(models_dir, "scaler.pkl")
        metadata_path = os.path.join(models_dir, "metadata.pkl")
        
        self.model = None
        self.scaler = None
        self.metadata = None
        
        # Load semua file jika tersedia
        if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(metadata_path):
            self.model = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            self.metadata = joblib.load(metadata_path)
            print("✅ BERHASIL: Model SVM, Scaler, dan Metadata telah diload!")
        else:
            print(f"❌ GAGAL: File .pkl tidak ditemukan di {models_dir}")

    def segmentCells(self, im_bgr):
        g_channel = im_bgr[:, :, 1]
        _, cell_mask = cv.threshold(g_channel, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
        contours, _ = cv.findContours(cell_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        return g_channel, cell_mask, contours

    def detectCentralPallor(self, im_bgr, g_channel, cell_mask, std_mult=1.2):
        g_pix = im_bgr[:, :, 1][cell_mask == 255]
        if len(g_pix) == 0:
            g_pix = np.array([128])
        batas = np.mean(g_pix) + std_mult * np.std(g_pix)
        _, cp_thresh = cv.threshold(g_channel, batas, 255, cv.THRESH_BINARY)
        cp_mask = cv.bitwise_and(cp_thresh, cp_thresh, mask=cell_mask)
        return cp_mask

    def pallorRatio(self, contours, cp_mask):
        if not contours: return 0.0
        largest_area = cv.contourArea(max(contours, key=cv.contourArea))
        cp_area = np.sum(cp_mask == 255)
        return cp_area / largest_area if largest_area > 0 else 0.0

    def ekstrak_fitur_satu_sel(self, img_bgr_pp):
        g_channel, cell_mask, contours = self.segmentCells(img_bgr_pp)

        area = perimeter = maj_ax = min_ax = 0
        compactness = eccentricity = solidity = aspect_ratio = 0
        rectangularity = convexity = circularity_ratio = euler_number = 0

        if contours:
            c = max(contours, key=cv.contourArea)
            area = cv.contourArea(c)
            perimeter = cv.arcLength(c, True)
            _, _, w_c, h_c = cv.boundingRect(c)
            bbox_area = w_c * h_c
            rectangularity = area / bbox_area if bbox_area > 0 else 0
            if len(c) >= 5:
                (_, _), (min_ax, maj_ax), _ = cv.fitEllipse(c)
                aspect_ratio = maj_ax / min_ax if min_ax > 0 else 0
                eccentricity = np.sqrt(1 - (min_ax**2 / maj_ax**2)) if maj_ax > min_ax else 0
            hull = cv.convexHull(c)
            hull_area = cv.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            convexity = cv.arcLength(hull, True) / perimeter if perimeter > 0 else 0
            compactness = (perimeter**2) / (4 * np.pi * area) if area > 0 else 0
            circularity_ratio = area / (perimeter**2) if perimeter > 0 else 0

        b_ch, g_ch, r_ch = cv.split(img_bgr_pp)
        mask_px = cell_mask == 255
        b_pix, g_pix, r_pix = b_ch[mask_px], g_ch[mask_px], r_ch[mask_px]
        if len(g_pix) == 0:
            b_pix = g_pix = r_pix = np.array([0])

        c_mean_r, c_std_r = np.mean(r_pix), np.std(r_pix)
        c_skew_r, c_kurt_r = float(skew(r_pix)), float(kurtosis(r_pix))
        c_mean_g, c_std_g = np.mean(g_pix), np.std(g_pix)
        c_skew_g, c_kurt_g = float(skew(g_pix)), float(kurtosis(g_pix))
        c_mean_b, c_std_b = np.mean(b_pix), np.std(b_pix)
        c_skew_b, c_kurt_b = float(skew(b_pix)), float(kurtosis(b_pix))

        cp_mask = self.detectCentralPallor(img_bgr_pp, g_channel, cell_mask, std_mult=1.2)
        pallor_ratio = self.pallorRatio(contours, cp_mask)
        
        cp_contours, _ = cv.findContours(cp_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        euler_number = 1 - sum(1 for cc in cp_contours if cv.contourArea(cc) > 10)

        cp_area = cp_perim = cp_maj = cp_min = cp_comp = cp_ecc = cp_solid = cp_ratio = 0
        if cp_contours:
            c_cp = max(cp_contours, key=cv.contourArea)
            cp_area = cv.contourArea(c_cp)
            cp_perim = cv.arcLength(c_cp, True)
            cp_ratio = pallor_ratio 
            if len(c_cp) >= 5:
                _, (cp_min, cp_maj), _ = cv.fitEllipse(c_cp)
                cp_ecc = np.sqrt(1 - (cp_min**2 / cp_maj**2)) if cp_maj > cp_min else 0
            hull_cp = cv.convexHull(c_cp)
            hull_cp_area = cv.contourArea(hull_cp)
            cp_solid = cp_area / hull_cp_area if hull_cp_area > 0 else 0
            cp_comp = (cp_perim**2) / (4 * np.pi * cp_area) if cp_area > 0 else 0

        pallor_px = cp_mask == 255
        rim_px = (cell_mask == 255) & (cp_mask == 0)
        r_pallor = r_ch[pallor_px].mean() if pallor_px.any() else 0
        r_rim = r_ch[rim_px].mean() if rim_px.any() else 0
        pallor_contrast = r_pallor - r_rim
        pallor_ratio_r = r_pallor / r_rim if r_rim > 0 else 0

        gray = cv.cvtColor(img_bgr_pp, cv.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)
        contrast = np.mean(graycoprops(glcm, 'contrast')[0])
        correlation = np.mean(graycoprops(glcm, 'correlation')[0])
        energy = np.mean(graycoprops(glcm, 'energy')[0])
        homogeneity = np.mean(graycoprops(glcm, 'homogeneity')[0])

        features = {
            "Area": round(area, 2), "Perimeter": round(perimeter, 2), "Major_Axis": round(maj_ax, 2),
            "Minor_Axis": round(min_ax, 2), "Compactness": round(compactness, 4), "Eccentricity": round(eccentricity, 4),
            "Solidity": round(solidity, 4), "Aspect_Ratio": round(aspect_ratio, 4), "Rectangularity": round(rectangularity, 4),
            "Convexity": round(convexity, 4), "Circularity_Ratio": round(circularity_ratio, 4), "Euler_Number": euler_number,
            "CP_Area": round(cp_area, 2), "CP_Perimeter": round(cp_perim, 2), "CP_Major_Axis": round(cp_maj, 2),
            "CP_Minor_Axis": round(cp_min, 2), "CP_Compactness": round(cp_comp, 4), "CP_Eccentricity": round(cp_ecc, 4),
            "CP_Solidity": round(cp_solid, 4), "CP_Ratio": round(cp_ratio, 4), "Pallor_Contrast_R": round(pallor_contrast, 4),
            "Pallor_Ratio_R": round(pallor_ratio_r, 4), "GLCM_Contrast_Mean": round(float(contrast), 6),
            "GLCM_Correlation_Mean": round(float(correlation), 6), "GLCM_Energy_Mean": round(float(energy), 6),
            "GLCM_Homogeneity_Mean": round(float(homogeneity), 6), "Color_Mean_R": round(c_mean_r, 4),
            "Color_Std_R": round(c_std_r, 4), "Color_Skewness_R": round(c_skew_r, 4), "Color_Kurtosis_R": round(c_kurt_r, 4),
            "Color_Mean_G": round(c_mean_g, 4), "Color_Std_G": round(c_std_g, 4), "Color_Skewness_G": round(c_skew_g, 4),
            "Color_Kurtosis_G": round(c_kurt_g, 4), "Color_Mean_B": round(c_mean_b, 4), "Color_Std_B": round(c_std_b, 4),
            "Color_Skewness_B": round(c_skew_b, 4), "Color_Kurtosis_B": round(c_kurt_b, 4),
        }
        return features

    def run_detection_pipeline(self, extracted_cells, bounding_boxes_sep, raw_image_rgb, output_dir, patient_id):
        if self.model is None or self.scaler is None or self.metadata is None:
            raise Exception("Model SVM tidak ditemukan! Pastikan file pkl ada di folder source/models/")

        selected_features = self.metadata.get('features', [])
        if not selected_features:
            raise Exception("Metadata tidak memiliki daftar 'features'.")

        result_img = cv.cvtColor(raw_image_rgb.copy(), cv.COLOR_RGB2BGR)
        ida_count = 0
        normal_count = 0

        for i, (cell_rgb, bbox) in enumerate(zip(extracted_cells, bounding_boxes_sep)):
            x, y, w, h = bbox
            cell_bgr = cv.cvtColor(cell_rgb, cv.COLOR_RGB2BGR)

            try:
                # 1. Ekstrak 40 Fitur
                features = self.ekstrak_fitur_satu_sel(cell_bgr)
                
                # 2. Ambil hanya fitur yang dipilih (Feature Selection)
                feature_vector = np.array([features[k] for k in selected_features]).reshape(1, -1)
                
                # 3. Normalisasi dengan Scaler
                feature_vector_scaled = self.scaler.transform(feature_vector)

                # 4. Prediksi (0 = Normal, 1 = IDA)
                prediction = self.model.predict(feature_vector_scaled)[0]

                if prediction == 1:
                    ida_count += 1
                    color = (255, 0, 0) # BIRU (BGR) untuk IDA
                else:
                    normal_count += 1
                    color = (0, 255, 0) # HIJAU (BGR) untuk Normal

                cv.rectangle(result_img, (x, y), (x + w, y + h), color, 5)

            except Exception as e:
                print(f"⚠️ Gagal memproses sebuah sel: {e}")

        # Simpan gambar deteksi
        res_path = os.path.join(output_dir, f"detection_result_{patient_id}.jpg")
        cv.imwrite(res_path, result_img)

        # Ambil Top 5 Fitur Terbaik dari metadata untuk ditampilkan
        top5 = selected_features[:5] if len(selected_features) >= 5 else selected_features
        return res_path, ida_count, normal_count, top5
