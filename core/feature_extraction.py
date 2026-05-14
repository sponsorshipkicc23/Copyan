import cv2 as cv
import numpy as np
import pandas as pd
import os
import warnings
from skimage.feature import graycomatrix, graycoprops
from scipy.stats import skew, kurtosis

warnings.filterwarnings("ignore")

def quality_filter_cells(cropped_cells, bounding_boxes, cell_masks, img_shape, border_margin=5, min_area=200, max_area=5000, min_dim=15, min_ar=0.6, max_ar=1.6):
    img_h, img_w = img_shape[:2]
    total_raw = len(cropped_cells)
    filtered_cells, filtered_boxes, filtered_masks, cell_labels = [], [], [], []
    stats = {"border": 0, "small": 0, "large": 0, "shape": 0, "dim": 0, "passed": 0}

    for idx in range(total_raw):
        x, y, w, h = bounding_boxes[idx]
        area = w * h
        original_label = idx + 1

        if (x <= border_margin or y <= border_margin or x + w >= img_w - border_margin or y + h >= img_h - border_margin):
            stats["border"] += 1
            continue
        if area < min_area:
            stats["small"] += 1
            continue
        if area > max_area:
            stats["large"] += 1
            continue
        ar = w / h if h > 0 else 0
        if ar < min_ar or ar > max_ar:
            stats["shape"] += 1
            continue
        if w < min_dim or h < min_dim:
            stats["dim"] += 1
            continue

        filtered_cells.append(cropped_cells[idx])
        filtered_boxes.append(bounding_boxes[idx])
        filtered_masks.append(cell_masks[idx])
        cell_labels.append(original_label)
        stats["passed"] += 1

    return filtered_cells, filtered_boxes, filtered_masks, cell_labels, stats

def extract_morphological_features(contour, mask):
    area = cv.contourArea(contour)
    perimeter = cv.arcLength(contour, True)

    if len(contour) >= 5:
        try:
            (_, _), (MA, ma), _ = cv.fitEllipse(contour)
            major_axis = max(MA, ma)
            minor_axis = min(MA, ma)
        except Exception:
            major_axis = minor_axis = 0.0
    else:
        major_axis = minor_axis = 0.0

    compactness = (perimeter ** 2) / (4 * np.pi * area) if area > 0 else 0.0
    eccentricity = (np.sqrt(1 - (minor_axis / major_axis) ** 2) if major_axis > 0 else 0.0)

    hull = cv.convexHull(contour)
    hull_area = cv.contourArea(hull)
    solidity = area / hull_area if hull_area > 0 else 0.0

    aspect_ratio = major_axis / minor_axis if minor_axis > 0 else 0.0

    return {
        "Area": area, "Perimeter": perimeter, "Major_Axis": major_axis, "Minor_Axis": minor_axis,
        "Compactness": compactness, "Eccentricity": eccentricity, "Solidity": solidity, "Aspect_Ratio": aspect_ratio,
    }

def extract_central_pallor_features(cell_img, cell_mask):
    gray = (cv.cvtColor(cell_img, cv.COLOR_RGB2GRAY) if len(cell_img.shape) == 3 else cell_img.copy())
    gray = cv.bitwise_and(gray, gray, mask=cell_mask)

    _, cp_binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    kernel = np.ones((3, 3), np.uint8)
    cp_binary = cv.erode(cp_binary, kernel, iterations=1)
    cp_contours, _ = cv.findContours(cp_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if len(cp_contours) > 0:
        cp_cnt = max(cp_contours, key=cv.contourArea)
        cp_area = cv.contourArea(cp_cnt)
        cp_perimeter = cv.arcLength(cp_cnt, True)
        if len(cp_cnt) >= 5:
            try:
                (_, _), (MA, ma), _ = cv.fitEllipse(cp_cnt)
                cp_major = max(MA, ma)
                cp_minor = min(MA, ma)
            except Exception:
                cp_major = cp_minor = 0.0
        else:
            cp_major = cp_minor = 0.0

        cp_compactness = ((cp_perimeter ** 2) / (4 * np.pi * cp_area) if cp_area > 0 else 0.0)
        cp_eccentricity = (np.sqrt(1 - (cp_minor / cp_major) ** 2) if cp_major > 0 else 0.0)
        cp_hull = cv.convexHull(cp_cnt)
        cp_hull_area = cv.contourArea(cp_hull)
        cp_solidity = cp_area / cp_hull_area if cp_hull_area > 0 else 0.0

        nonzero = cv.findNonZero(cell_mask)
        cell_area_px = cv.contourArea(nonzero) if nonzero is not None else 0
        cp_ratio = cp_area / cell_area_px if cell_area_px > 0 else 0.0
    else:
        cp_area = cp_perimeter = cp_major = cp_minor = 0.0
        cp_compactness = cp_eccentricity = cp_solidity = cp_ratio = 0.0

    return {
        "CP_Area": cp_area, "CP_Perimeter": cp_perimeter, "CP_Major_Axis": cp_major, "CP_Minor_Axis": cp_minor,
        "CP_Compactness": cp_compactness, "CP_Eccentricity": cp_eccentricity, "CP_Solidity": cp_solidity, "CP_Ratio": cp_ratio,
    }

def extract_glcm_features(cell_img, cell_mask):
    gray = (cv.cvtColor(cell_img, cv.COLOR_RGB2GRAY) if len(cell_img.shape) == 3 else cell_img.copy())
    gray = cv.bitwise_and(gray, gray, mask=cell_mask)
    gray_q = (gray // 16).astype(np.uint8)

    distances = [1]
    angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
    angle_labels = ["0", "45", "90", "135"]

    try:
        glcm = graycomatrix(gray_q, distances=distances, angles=angles, levels=16, symmetric=True, normed=True)
        props = {
            "Contrast": graycoprops(glcm, "contrast")[0], "Correlation": graycoprops(glcm, "correlation")[0],
            "Energy": graycoprops(glcm, "energy")[0], "Homogeneity": graycoprops(glcm, "homogeneity")[0],
        }
        features = {}
        for prop_name, values in props.items():
            for i, lbl in enumerate(angle_labels):
                features[f"GLCM_{prop_name}_{lbl}"] = float(values[i])
            features[f"GLCM_{prop_name}_Mean"] = float(np.mean(values))
    except Exception:
        features = {}
        for prop_name in ["Contrast", "Correlation", "Energy", "Homogeneity"]:
            for lbl in angle_labels + ["Mean"]:
                features[f"GLCM_{prop_name}_{lbl}"] = 0.0

    return features

def extract_color_moment_features(cell_img, cell_mask):
    features = {}
    channels = ["R", "G", "B"]

    if len(cell_img.shape) == 3:
        for i, ch in enumerate(channels):
            channel = cell_img[:, :, i]
            masked = cv.bitwise_and(channel, channel, mask=cell_mask)
            pixels = masked[cell_mask > 0]
            if len(pixels) > 0:
                features[f"Color_Mean_{ch}"] = float(np.mean(pixels))
                features[f"Color_Std_{ch}"] = float(np.std(pixels))
                features[f"Color_Skewness_{ch}"] = float(skew(pixels))
                features[f"Color_Kurtosis_{ch}"] = float(kurtosis(pixels))
            else:
                for moment in ["Mean", "Std", "Skewness", "Kurtosis"]:
                    features[f"Color_{moment}_{ch}"] = 0.0
    else:
        for ch in channels:
            for moment in ["Mean", "Std", "Skewness", "Kurtosis"]:
                features[f"Color_{moment}_{ch}"] = 0.0

    return features

def extract_all_features(cell_img, cell_mask, cell_label, bbox_coords=None):
    contours, _ = cv.findContours(cell_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0: return None

    contour = max(contours, key=cv.contourArea)
    features = {"Cell_Label": cell_label}

    if bbox_coords is not None:
        features["X"], features["Y"] = bbox_coords[0], bbox_coords[1]
    else:
        features["X"], features["Y"] = 0, 0

    features.update(extract_morphological_features(contour, cell_mask))
    features.update(extract_central_pallor_features(cell_img, cell_mask))
    features.update(extract_glcm_features(cell_img, cell_mask))
    features.update(extract_color_moment_features(cell_img, cell_mask))

    return features

def run_feature_extraction(extracted_cells, bounding_boxes, cell_masks, img_shape, output_csv_path=None, border_margin=5, min_area=200, max_area=5000, min_dim=15, min_ar=0.6, max_ar=1.6):
    cell_imgs = [item[0] if isinstance(item, tuple) else item for item in extracted_cells]

    filtered_cells, filtered_boxes, filtered_masks, cell_labels, filter_stats = \
        quality_filter_cells(cell_imgs, bounding_boxes, cell_masks, img_shape, border_margin=border_margin, min_area=min_area, max_area=max_area, min_dim=min_dim, min_ar=min_ar, max_ar=max_ar)

    all_features = []
    for idx in range(len(filtered_cells)):
        feat = extract_all_features(filtered_cells[idx], filtered_masks[idx], cell_label=cell_labels[idx], bbox_coords=filtered_boxes[idx])
        if feat is not None: all_features.append(feat)

    if len(all_features) == 0:
        df_features = pd.DataFrame()
    else:
        df_features = pd.DataFrame(all_features).sort_values("Cell_Label").reset_index(drop=True)

    if output_csv_path is not None and not df_features.empty:
        os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
        df_features.to_csv(output_csv_path, index=False)

    return df_features, cell_labels, filter_stats
