import cv2 as cv
import numpy as np
import time
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from skimage.feature import peak_local_max

def convert_hsv_circular(image_rgb, v_thresh=20):
    hsv_image = cv.cvtColor(image_rgb, cv.COLOR_RGB2HSV)
    v = hsv_image[:, :, 2]
    mask = v > v_thresh
    return hsv_image, mask

def apply_median_filter(img_rgb, kernel_size=3):
    return cv.medianBlur(img_rgb, kernel_size)

# --- FUNGSI REINHARD NORMALIZATION ---
def reinhard_normalization(Source, Target, epsilon=1e-6):
    src = cv.cvtColor(Source, cv.COLOR_RGB2LAB).astype(float)
    tgt = cv.cvtColor(Target, cv.COLOR_RGB2LAB).astype(float)
    result = []

    for i in range(3):
        src_channel = src[:, :, i]
        tgt_channel = tgt[:, :, i]

        src_mean, src_std = np.mean(src_channel), np.std(src_channel)
        tgt_mean, tgt_std = np.mean(tgt_channel), np.std(tgt_channel)

        # Normalisasi
        normalized = (src_channel - src_mean) * (tgt_std / (src_std + epsilon)) + tgt_mean
        result.append(normalized)

    # Wajib di clip agar tidak overflow warna (titik-titik noise)
    merged = np.clip(cv.merge(result), 0, 255).astype(np.uint8)
    return cv.cvtColor(merged, cv.COLOR_LAB2RGB)

def apply_clahe(img_rgb, clip_limit=2.0, tile_grid_size=(8, 8)):
    lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2LAB)
    L, A, B = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    L_cl = clahe.apply(L)
    lab_cl = cv.merge((L_cl, A, B))
    return cv.cvtColor(lab_cl, cv.COLOR_LAB2RGB)

def apply_log_enhancement(img_clahe, sigma=1.5, alpha=0.5):
    img_clahe_float = img_clahe.astype(np.float32) / 255.0
    ksize = int(2 * np.ceil(3 * sigma) + 1)
    img_log_edges = np.zeros_like(img_clahe_float)
    for i in range(3):
        blurred = cv.GaussianBlur(img_clahe_float[:,:,i], (ksize, ksize), sigma)
        laplacian = cv.Laplacian(blurred, cv.CV_32F, ksize=3)
        img_log_edges[:,:,i] = laplacian
    img_sharpened = img_clahe_float - alpha * img_log_edges
    img_sharpened = np.clip(img_sharpened, 0, 1)
    return (img_sharpened * 255).astype(np.uint8)

def preprocess_image(img_rgb, ref_img_rgb=None):
    # 1. Median Filter
    img_denoised = apply_median_filter(img_rgb, kernel_size=3)
    
    # 2. Reinhard Normalization (Jika ada gambar referensi)
    if ref_img_rgb is not None:
        img_norm = reinhard_normalization(img_denoised, ref_img_rgb)
    else:
        img_norm = img_denoised
        
    # 3. CLAHE
    img_clahe = apply_clahe(img_norm, clip_limit=2.0, tile_grid_size=(8, 8))
    
    # 4. LoG Enhancement
    img_preprocessed = apply_log_enhancement(img_clahe, sigma=1.5, alpha=0.5)
    return img_preprocessed

def kmeans_segmentation(image, k, use_preprocessing=True, v_thresh=20, ref_img_rgb=None):
    if use_preprocessing:
        img_rgb = cv.cvtColor(image, cv.COLOR_HSV2RGB)
        img_preprocessed = preprocess_image(img_rgb, ref_img_rgb=ref_img_rgb)
    else:
        img_rgb = cv.cvtColor(image, cv.COLOR_HSV2RGB)
        img_preprocessed = img_rgb
    
    hsv_preprocessed = cv.cvtColor(img_preprocessed, cv.COLOR_RGB2HSV)
    pixels = hsv_preprocessed.reshape(-1, 3)
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    
    segmented_images = []
    for i in range(k):
        cluster_image = np.zeros_like(pixels)
        cluster_image[labels == i] = centers[i]
        segmented_image = cluster_image.reshape(hsv_preprocessed.shape)
        segmented_image = np.clip(segmented_image, 0, 255).astype(np.uint8)
        segmented_images.append(segmented_image)
    
    return segmented_images, labels

def remove_unwanted_cells_extended(clustered_images, selected_cluster, original_image):
    if not selected_cluster: raise ValueError("No clusters selected.")
    segmented_mask = clustered_images[selected_cluster[0]].copy()
    for index_cluster in selected_cluster[1:]:
        segmented_mask = cv.bitwise_or(segmented_mask, clustered_images[index_cluster])
    
    segmented_mask = cv.cvtColor(segmented_mask, cv.COLOR_HSV2RGB)
    segmented_mask = cv.cvtColor(segmented_mask, cv.COLOR_RGB2GRAY)
    _, binary_mask = cv.threshold(segmented_mask, 1, 255, cv.THRESH_BINARY)

    rbc_segment = cv.bitwise_and(original_image, original_image, mask=binary_mask)
    rbc_segment_gray = cv.cvtColor(rbc_segment, cv.COLOR_RGB2GRAY)

    kernel_open = np.ones((5, 5), np.uint8)
    kernel_close = np.ones((5, 5), np.uint8)
    rbc_segment_gray = cv.morphologyEx(rbc_segment_gray, cv.MORPH_OPEN, kernel_open)
    rbc_segment_gray = cv.morphologyEx(rbc_segment_gray, cv.MORPH_CLOSE, kernel_close)
    
    contours, _ = cv.findContours(rbc_segment_gray, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    MIN_AREA = 120
    filtered_mask = np.zeros_like(rbc_segment_gray)
    for contour in contours:
        if cv.contourArea(contour) >= MIN_AREA:
            cv.drawContours(filtered_mask, [contour], -1, 255, thickness=cv.FILLED)

    rbc_only_image = cv.bitwise_and(rbc_segment, rbc_segment, mask=filtered_mask)
    for c in range(rbc_only_image.shape[2]):
        _, rbc_only_image[:, :, c] = cv.threshold(rbc_only_image[:, :, c], 15, 255, cv.THRESH_TOZERO)
    
    return rbc_only_image, filtered_mask, binary_mask

def bounded_opening_frs(binary_mask, num_openings=3):
    kernel_size = 5
    processed_mask = binary_mask.copy()
    for iteration in range(num_openings):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        processed_mask = cv.morphologyEx(processed_mask, cv.MORPH_OPEN, kernel)
        kernel_size += 2
    
    dist_transform = cv.distanceTransform(processed_mask, cv.DIST_L2, 5)
    dist_norm = cv.normalize(dist_transform, None, 0, 1.0, cv.NORM_MINMAX)
    
    radii = [5, 7, 9, 11, 13]
    frs_maps = []
    for radius in radii:
        grad_x = cv.Sobel(dist_norm, cv.CV_64F, 1, 0, ksize=3)
        grad_y = cv.Sobel(dist_norm, cv.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        orientation = np.arctan2(grad_y, grad_x)
        symmetry_map = np.zeros_like(dist_norm)
        y_coords, x_coords = np.where(processed_mask > 0)
        if len(y_coords) > 0:
            angles = orientation[y_coords, x_coords]
            grads = grad_mag[y_coords, x_coords]
            px = (x_coords + radius * np.cos(angles)).astype(int)
            py = (y_coords + radius * np.sin(angles)).astype(int)
            nx = (x_coords - radius * np.cos(angles)).astype(int)
            ny = (y_coords - radius * np.sin(angles)).astype(int)
            valid_p = (px >= 0) & (px < symmetry_map.shape[1]) & (py >= 0) & (py < symmetry_map.shape[0])
            valid_n = (nx >= 0) & (nx < symmetry_map.shape[1]) & (ny >= 0) & (ny < symmetry_map.shape[0])
            np.add.at(symmetry_map, (py[valid_p], px[valid_p]), grads[valid_p])
            np.add.at(symmetry_map, (ny[valid_n], nx[valid_n]), grads[valid_n])
        frs_maps.append(symmetry_map)
    
    frs_combined = np.mean(frs_maps, axis=0)
    frs_combined = cv.normalize(frs_combined, None, 0, 1.0, cv.NORM_MINMAX)
    combined_map = 0.6 * dist_norm + 0.4 * frs_combined
    
    rough_coords = peak_local_max(combined_map, min_distance=6, threshold_abs=0.08, exclude_border=False)
    if len(rough_coords) > 0:
        rough_radii = [dist_transform[y, x] for y, x in rough_coords]
        rough_median_radius = np.median(rough_radii)
        adaptive_min_dist = max(8, int(rough_median_radius * 1.4))
    else:
        adaptive_min_dist = 10
    
    coordinates = peak_local_max(combined_map, min_distance=adaptive_min_dist, threshold_abs=0.1, exclude_border=False)
    centers = [(int(x), int(y)) for y, x in coordinates]
    
    radii_list = []
    for (cx, cy) in centers:
        if 0 <= cx < dist_transform.shape[1] and 0 <= cy < dist_transform.shape[0]:
            radii_list.append(dist_transform[cy, cx])
    
    if len(radii_list) > 0:
        candidate_radius = int(np.median(radii_list))
        radius_std = np.std(radii_list)
        min_acceptable_radius = candidate_radius * 0.5
        max_acceptable_radius = candidate_radius * 1.8
        filtered_centers = []
        for (cx, cy), r in zip(centers, radii_list):
            if min_acceptable_radius <= r <= max_acceptable_radius:
                filtered_centers.append((cx, cy))
        centers = filtered_centers
    else:
        candidate_radius = 15
        radius_std = 0
    
    center_map = np.zeros_like(processed_mask)
    for (cx, cy) in centers: cv.circle(center_map, (cx, cy), 3, 255, -1)
    
    return {
        'refined_mask': processed_mask, 'dist_transform': dist_transform, 'frs_map': frs_combined,
        'combined_map': combined_map, 'centers': centers, 'center_map': center_map,
        'candidate_radius': candidate_radius, 'radius_std': radius_std
    }

def separate_overlapping_rbc_with_gmm(bofrs_results, cells_image):
    centers_global = bofrs_results['centers']
    dist_transform = bofrs_results['dist_transform']
    refined_mask = bofrs_results['refined_mask']
    candidate_radius = bofrs_results['candidate_radius']
    
    all_cropped_cells = []
    all_bounding_boxes = []
    all_cell_masks = []
    
    contours, _ = cv.findContours(refined_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    min_area = max(100, int(np.pi * (candidate_radius * 0.6) ** 2))
    max_area_for_single = int(np.pi * candidate_radius ** 2) * 1.8
    
    for idx, contour in enumerate(contours):
        x_offset, y_offset, w, h = cv.boundingRect(contour)
        area = cv.contourArea(contour)
        if w < 10 or h < 10 or area < min_area: continue
        
        single_mask = np.zeros_like(refined_mask)
        cv.drawContours(single_mask, [contour], -1, 255, thickness=cv.FILLED)
        
        cropped_mask = single_mask[y_offset:y_offset+h, x_offset:x_offset+w]
        cropped_image = cells_image[y_offset:y_offset+h, x_offset:x_offset+w]
        cropped_dist = dist_transform[y_offset:y_offset+h, x_offset:x_offset+w]
        
        local_centers = []
        for (cx, cy) in centers_global:
            if x_offset <= cx < x_offset+w and y_offset <= cy < y_offset+h:
                local_centers.append((cx - x_offset, cy - y_offset))
        
        k = len(local_centers)
        if (k == 0 and area <= max_area_for_single) or (k == 1 and area <= max_area_for_single):
            all_bounding_boxes.append((x_offset, y_offset, w, h))
            all_cropped_cells.append(cropped_image)
            all_cell_masks.append(cropped_mask)
            continue
        
        if area <= max_area_for_single and k > 1:
            min_dist_between_centers = candidate_radius * 1.2
            valid_centers = [local_centers[0]]
            for center in local_centers[1:]:
                if all(np.sqrt((center[0] - vc[0])**2 + (center[1] - vc[1])**2) >= min_dist_between_centers for vc in valid_centers):
                    valid_centers.append(center)
            if len(valid_centers) == 1:
                all_bounding_boxes.append((x_offset, y_offset, w, h))
                all_cropped_cells.append(cropped_image)
                all_cell_masks.append(cropped_mask)
                continue
            else:
                k = len(valid_centers)
                local_centers = valid_centers
        
        X_replicated = []
        for (lx, ly) in local_centers:
            if 0 <= lx < cropped_dist.shape[1] and 0 <= ly < cropped_dist.shape[0]:
                weight = min(int(cropped_dist[ly, lx]), max(20, int(candidate_radius * 2)))
                X_replicated.extend([(lx, ly)] * max(1, weight))
        
        if len(X_replicated) < k * 10:
            all_bounding_boxes.append((x_offset, y_offset, w, h))
            all_cropped_cells.append(cropped_image)
            all_cell_masks.append(cropped_mask)
            continue
        
        try:
            gmm = GaussianMixture(n_components=k, covariance_type="tied", max_iter=100, n_init=3, random_state=42, tol=1e-3)
            gmm.fit(np.array(X_replicated))
            ys, xs = np.where(cropped_mask == 255)
            if len(ys) == 0: continue
            
            foreground_pixels = np.column_stack((xs, ys))
            pixel_labels = gmm.predict(foreground_pixels)
            
            labeled_mask = np.zeros_like(cropped_mask, dtype=np.uint8)
            for (x, y), label in zip(foreground_pixels, pixel_labels):
                labeled_mask[y, x] = label + 1
            
            unique_labels = np.unique(labeled_mask)
            unique_labels = unique_labels[unique_labels != 0]
            
            for label in unique_labels:
                cell_mask = (labeled_mask == label).astype(np.uint8) * 255
                cell_image = cv.bitwise_and(cropped_image, cropped_image, mask=cell_mask)
                coords = cv.findNonZero(cell_mask)
                if coords is not None and len(coords) > 50:
                    x_cell, y_cell, w_cell, h_cell = cv.boundingRect(coords)
                    global_x = x_cell + x_offset
                    global_y = y_cell + y_offset
                    cell_img_cropped = cell_image[y_cell:y_cell+h_cell, x_cell:x_cell+w_cell]
                    cell_mask_cropped = cell_mask[y_cell:y_cell+h_cell, x_cell:x_cell+w_cell]
                    
                    all_bounding_boxes.append((global_x, global_y, w_cell, h_cell))
                    all_cropped_cells.append(cell_img_cropped)
                    all_cell_masks.append(cell_mask_cropped)
        except Exception:
            all_bounding_boxes.append((x_offset, y_offset, w, h))
            all_cropped_cells.append(cropped_image)
            all_cell_masks.append(cropped_mask)
            
    return all_cropped_cells, all_bounding_boxes, all_cell_masks

def sobel_edge_detect(image):
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=5)
    sobel_edges = cv.magnitude(sobel_x, sobel_y)
    sobel_edges = np.uint8(255 * (sobel_edges / np.max(sobel_edges)))
    _, sobel_binary = cv.threshold(sobel_edges, 50, 255, cv.THRESH_BINARY)
    contours_sobel, _ = cv.findContours(sobel_binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    return sobel_edges, contours_sobel

def draw_bounding_boxes(image, contours):
    bbox_image = image.copy()
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)
        cv.rectangle(bbox_image, (x, y), (x + w, y + h), (0, 255, 0), 5)
    return bbox_image

def extract_contours(image, edge_map):
    contours, _ = cv.findContours(edge_map, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contour_mask = np.zeros_like(image)
    cv.drawContours(contour_mask, contours, -1, 255, thickness=cv.FILLED)
    return contours, contour_mask
