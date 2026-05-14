import os
import cv2 as cv
import numpy as np
import imageio
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any

from sklearn.feature_selection import mutual_info_classif

# Local modules (unchanged algorithmic code)
from Segmentation import (
    convert_hsv_circular,
    kmeans_segmentation,
    remove_unwanted_cells_extended,
    bounded_opening,
    bounded_opening_frs,
    separate_overlapping_rbc_with_gmm,
    sobel_edge_detect,
    draw_bounding_boxes,
    extract_contours,
)
from Feature_Extraction import run_feature_extraction as _run_feature_extraction

CellTuple  = Tuple[np.ndarray, int, int]          # (crop, x, y)
BBox       = Tuple[int, int, int, int]             # (x, y, w, h)
CellInfo   = Dict[str, Any]                        # {"filename": …, "bbox": […]}


# K-Means
@dataclass
class SegmentationResult:
    raw_image:       np.ndarray
    hsv_clean_image: np.ndarray
    segmented_images: List[np.ndarray]
    labels_full:     np.ndarray


def run_segmentation(image_path: str, k: int = 6) -> SegmentationResult:
    raw_image = cv.cvtColor(
        imageio.imread(image_path).astype(np.uint8), cv.COLOR_BGR2RGB
    )
    hsv_clean_image, _ = convert_hsv_circular(raw_image, v_thresh=20)

    segmented_images, labels_full = kmeans_segmentation(
        hsv_clean_image, k=k, use_preprocessing=True, v_thresh=20
    )

    return SegmentationResult(
        raw_image=raw_image,
        hsv_clean_image=hsv_clean_image,
        segmented_images=segmented_images,
        labels_full=labels_full,
    )


# Cell Extraction
@dataclass
class ExtractionResult:
    rbc_only_image:    np.ndarray
    filtered_mask:     np.ndarray
    annotated_image:   np.ndarray          # with numbered bounding boxes
    extracted_cells:   List[CellTuple]
    cell_masks_list:   List[np.ndarray]
    bounding_boxes:    List[BBox]


def run_cell_extraction(
    seg_result: SegmentationResult,
    selected_clusters: List[int],
) -> ExtractionResult:
    rgb_clean = cv.cvtColor(seg_result.hsv_clean_image, cv.COLOR_HSV2RGB)

    rbc_only_image, filtered_mask, _ = remove_unwanted_cells_extended(
        seg_result.segmented_images, selected_clusters, rgb_clean
    )

    gray = cv.cvtColor(rbc_only_image, cv.COLOR_RGB2GRAY)
    edge_map, contour_edge = sobel_edge_detect(gray)

    annotated = draw_bounding_boxes(rbc_only_image, contour_edge)
    annotated  = _add_cell_numbers(annotated, contour_edge, gray, edge_map)

    contours, _ = extract_contours(gray, edge_map)
    extracted_cells, cell_masks_list, bounding_boxes = _crop_cells(
        rbc_only_image, contours
    )

    return ExtractionResult(
        rbc_only_image=rbc_only_image,
        filtered_mask=filtered_mask,
        annotated_image=annotated,
        extracted_cells=extracted_cells,
        cell_masks_list=cell_masks_list,
        bounding_boxes=bounding_boxes,
    )


def _add_cell_numbers(
    image: np.ndarray, contours, gray, edge_map
) -> np.ndarray:
    """Overlay numbered labels on bounding boxes."""
    annotated = image.copy()
    real_contours, _ = extract_contours(gray, edge_map)
    font, scale, thick = cv.FONT_HERSHEY_SIMPLEX, 0.6, 2

    for idx, contour in enumerate(real_contours, start=1):
        x, y, w, h = cv.boundingRect(contour)
        cx, cy = x + w // 2, y + h // 2
        (lw, lh), _ = cv.getTextSize(str(idx), font, scale, thick)
        pad = 4
        lx = max(0, min(cx - lw // 2 - pad,
                        annotated.shape[1] - lw - 2 * pad))
        ly = max(lh + 2 * pad,
                 min(cy - lh // 2 - pad, annotated.shape[0]))
        cv.rectangle(annotated,
                     (lx, ly - lh - pad), (lx + lw + 2 * pad, ly + pad),
                     (0, 0, 0), -1)
        cv.putText(annotated, str(idx),
                   (lx + pad, ly - pad), font, scale,
                   (255, 255, 255), thick, cv.LINE_AA)
    return annotated


def _crop_cells(
    image: np.ndarray, contours
) -> Tuple[List[CellTuple], List[np.ndarray], List[BBox]]:
    """Crop each contour into an individual cell image + mask."""
    extracted_cells: List[CellTuple] = []
    cell_masks_list: List[np.ndarray] = []
    bounding_boxes:  List[BBox]       = []

    for contour in contours:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv.drawContours(mask, [contour], -1, 255, -1)
        x, y, w, h = cv.boundingRect(contour)
        cell = cv.bitwise_and(
            image[y:y+h, x:x+w],
            image[y:y+h, x:x+w],
            mask=mask[y:y+h, x:x+w],
        )
        extracted_cells.append((cell, x, y))
        cell_masks_list.append(mask[y:y+h, x:x+w])
        bounding_boxes.append((x, y, w, h))

    return extracted_cells, cell_masks_list, bounding_boxes


# BO-FRS + GMM
@dataclass
class SeparationResult:
    extracted_cells:  List[CellTuple]
    cell_info:        List[CellInfo]
    cell_masks_list:  List[np.ndarray]
    bounding_boxes:   List[BBox]
    annotated_image:  np.ndarray


def run_overlap_separation(
    ext_result: ExtractionResult,
) -> SeparationResult:
    """
    Apply BO-FRS + GMM to separate touching/overlapping RBCs.
    """
    opened_mask  = bounded_opening(ext_result.filtered_mask, num_openings=3)
    bofrs_result = bounded_opening_frs(opened_mask, num_openings=3)

    cropped_cells, bboxes, cell_masks = separate_overlapping_rbc_with_gmm(
        bofrs_result, ext_result.rbc_only_image
    )

    extracted_cells: List[CellTuple] = []
    cell_info:       List[CellInfo]  = []
    cell_masks_list: List[np.ndarray] = []

    for idx, (cell_img, bbox) in enumerate(zip(cropped_cells, bboxes)):
        x, y, w, h = bbox
        extracted_cells.append((cell_img, x, y))
        cell_info.append({"filename": f"cell_{idx}.png", "bbox": list(bbox)})
        cell_masks_list.append(cell_masks[idx])

    annotated = _annotate_bboxes(ext_result.rbc_only_image, bboxes)

    return SeparationResult(
        extracted_cells=extracted_cells,
        cell_info=cell_info,
        cell_masks_list=cell_masks_list,
        bounding_boxes=bboxes,
        annotated_image=annotated,
    )


def _annotate_bboxes(
    image: np.ndarray, bboxes: List[BBox]
) -> np.ndarray:
    """Draw numbered green bounding boxes on a copy of image."""
    annotated = image.copy()
    font, scale, thick = cv.FONT_HERSHEY_SIMPLEX, 0.6, 2

    for idx, (x, y, w, h) in enumerate(bboxes, start=1):
        cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 5)
        (lw, lh), _ = cv.getTextSize(str(idx), font, scale, thick)
        pad = 4
        cx, cy = x + w // 2, y + h // 2
        lx = max(0, min(cx - lw // 2 - pad,
                        annotated.shape[1] - lw - 2 * pad))
        ly = max(lh + 2 * pad,
                 min(cy - lh // 2 - pad, annotated.shape[0]))
        cv.rectangle(annotated,
                     (lx, ly - lh - pad), (lx + lw + 2 * pad, ly + pad),
                     (0, 0, 0), -1)
        cv.putText(annotated, str(idx),
                   (lx + pad, ly - pad), font, scale,
                   (255, 255, 255), thick, cv.LINE_AA)
    return annotated


# Feature Extraction + Save
@dataclass
class FeatureResult:
    df_features:  pd.DataFrame
    filter_stats: Dict[str, int]
    excel_path:   str


def run_save_features(
    sep_result: SeparationResult,
    rbc_only_image: np.ndarray,
    excel_path: str,
) -> FeatureResult:
    """
    Quality-filter cells, extract all features, save to Excel.
    """
    df, _, filter_stats = _run_feature_extraction(
        extracted_cells=sep_result.extracted_cells,
        bounding_boxes=sep_result.bounding_boxes,
        cell_masks=sep_result.cell_masks_list,
        img_shape=rbc_only_image.shape,
        output_csv_path=None,
    )

    if not df.empty:
        df.to_excel(excel_path, index=False)

    return FeatureResult(
        df_features=df,
        filter_stats=filter_stats,
        excel_path=excel_path,
    )


# Feature Selection
@dataclass
class DetectionResult:
    df_selected:     pd.DataFrame
    mi_df:           pd.DataFrame
    ida_count:       int
    normal_count:    int
    top_features:    List[str]
    annotated_image: np.ndarray
    selected_path:   str
    mi_path:         str


_IDA_COLOR   = (0, 0, 255)    # red   – IDA / abnormal
_NORMAL_COLOR = (0, 255, 0)   # green – normal
_UNKNOWN_COLOR = (128, 128, 128)

_COLOR_MAP = {0: _NORMAL_COLOR, 1: _IDA_COLOR, -1: _UNKNOWN_COLOR}


def run_detection(
    feat_result: FeatureResult,
    cell_info: List[CellInfo],
    raw_image: np.ndarray,
    selected_path: str,
    mi_path: str,
) -> DetectionResult:
    """
    1. Rule-based IDA labelling (area + CP_Ratio thresholds).
    2. Mutual information feature selection.
    3. Annotate the raw image with colour-coded bounding boxes.
    """
    df = feat_result.df_features.copy()

    # ── Rule-based labelling ──────────────────────────────────────────────────
    area_thresh = df["Area"].quantile(0.33)
    cp_thresh   = df["CP_Ratio"].quantile(0.67)
    df["IDA_Label"] = (
        (df["Area"] < area_thresh) & (df["CP_Ratio"] > cp_thresh)
    ).astype(int)

    ida_count    = int(df["IDA_Label"].sum())
    normal_count = len(df) - ida_count

    # ── Mutual information ────────────────────────────────────────────────────
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
    selected_features = mi_df[mi_df["MI_Score"] > 0.01]["Feature"].tolist()

    df_selected = df[["Cell_Label", "X", "Y"] + selected_features + ["IDA_Label"]]

    # ── Save tables ───────────────────────────────────────────────────────────
    df_selected.to_excel(selected_path, index=False)
    mi_df.to_excel(mi_path, index=False)

    # ── Annotate image ────────────────────────────────────────────────────────
    annotated = raw_image.copy()
    for row_idx, info in enumerate(cell_info):
        x, y, w, h = info["bbox"]
        row_data = df[df["Cell_Label"] == row_idx + 1]
        label = int(row_data["IDA_Label"].values[0]) if not row_data.empty else -1
        cv.rectangle(annotated, (x, y), (x + w, y + h),
                     _COLOR_MAP[label], 5)

    return DetectionResult(
        df_selected=df_selected,
        mi_df=mi_df,
        ida_count=ida_count,
        normal_count=normal_count,
        top_features=mi_df.head(5)["Feature"].tolist(),
        annotated_image=annotated,
        selected_path=selected_path,
        mi_path=mi_path,
    )
