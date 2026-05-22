import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. Project path
# ============================================================

PROJECT_DIR = Path(
    r"D:\M.Sc. Autonomous Systems - DTU\Spring Semester\34241 Digital video technology\Project"
)


# ============================================================
# 2. Video settings
# ============================================================

VIDEOS = {
    "DJI_1": {
        "video_type": "thermal",
        "video_path": PROJECT_DIR / "DJI_1.mp4",

        # Same folders as DJI_1 v2 analysis
        "frame_dir": PROJECT_DIR / "v2_frames_DJI_1",
        "crop_dir": PROJECT_DIR / "v2_cropped_frames_DJI_1",
        "mask_dir": PROJECT_DIR / "v2_panel_masks_DJI_1",
        "hotspot_mask_dir": PROJECT_DIR / "v2_hotspot_masks_DJI_1",
        "overlay_dir": PROJECT_DIR / "v2_hotspot_overlays_DJI_1",
        "result_dir": PROJECT_DIR / "v2_results_DJI_1",
        "plot_dir": PROJECT_DIR / "v2_plots_DJI_1",

        # Thermal crop used earlier
        "crop": (0, 135, 640, 410),

        # Fixed thresholds for thermal video
        "sharpness_threshold": 800,
        "panel_coverage_threshold": 0.70,
        "junction_threshold": 0.15,
        "hotspot_contrast_threshold": 30,

        # Sampling
        "sample_seconds": 1.5,
    },

    "DJI_2": {
        "video_type": "rgb",
        "video_path": PROJECT_DIR / "DJI_2.mp4",

        # Same folders as current DJI_2 analysis
        "frame_dir": PROJECT_DIR / "frames_DJI_2",
        "crop_dir": PROJECT_DIR / "cropped_frames_DJI_2",
        "mask_dir": PROJECT_DIR / "panel_masks_DJI_2",
        "hotspot_mask_dir": PROJECT_DIR / "hotspot_masks_DJI_2",
        "overlay_dir": PROJECT_DIR / "hotspot_overlays_DJI_2",
        "result_dir": PROJECT_DIR / "results_DJI_2",
        "plot_dir": PROJECT_DIR / "plots_DJI_2",

        # Your latest working RGB crop
        "crop": (0, 690, 2048, 1800),

        # Adaptive thresholds are used for RGB later
        "junction_threshold": 0.12,

        # Sampling
        "sample_seconds": 2.0,
    },
}


# ============================================================
# 3. Common helper functions
# ============================================================

def create_dirs(cfg):
    for key in [
        "frame_dir",
        "crop_dir",
        "mask_dir",
        "hotspot_mask_dir",
        "overlay_dir",
        "result_dir",
        "plot_dir",
    ]:
        cfg[key].mkdir(exist_ok=True)


def sharpness_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def sobel_sharpness(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(gx ** 2 + gy ** 2))


def estimate_panel_coverage_thermal(panel_crop):
    """
    Thermal DJI_1 panel mask.
    Keeps orange/yellow thermal panel regions.
    """

    hsv = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2HSV)

    lower_panel = np.array([5, 70, 80])
    upper_panel = np.array([45, 255, 255])

    panel_mask = cv2.inRange(hsv, lower_panel, upper_panel)

    kernel = np.ones((5, 5), np.uint8)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_OPEN, kernel)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, kernel)

    panel_coverage = np.sum(panel_mask > 0) / panel_mask.size

    return panel_coverage, panel_mask


def estimate_panel_coverage_rgb(panel_crop):
    """
    RGB DJI_2 panel mask.
    Keeps dark grey/blue panel regions and removes green grass.
    """

    hsv = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    green_mask = (
        (h >= 35) & (h <= 90) &
        (s >= 35) &
        (v >= 50)
    )

    dark_panel_mask = (
        (v <= 170) &
        (~green_mask)
    )

    panel_mask = dark_panel_mask.astype(np.uint8) * 255

    kernel = np.ones((7, 7), np.uint8)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_OPEN, kernel)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, kernel)

    panel_coverage = np.sum(panel_mask > 0) / panel_mask.size

    return panel_coverage, panel_mask


def detect_vertical_junction(panel_mask, threshold):
    """
    Rejects large vertical non-panel gaps.
    Useful for gaps between arrays, support structures, or central junctions.
    """

    h, w = panel_mask.shape
    binary = panel_mask > 0

    column_panel_fraction = np.mean(binary, axis=0)
    non_panel_columns = column_panel_fraction < 0.25

    max_gap_width = 0
    current_gap_width = 0

    for value in non_panel_columns:
        if value:
            current_gap_width += 1
            max_gap_width = max(max_gap_width, current_gap_width)
        else:
            current_gap_width = 0

    max_gap_ratio = max_gap_width / w
    junction_detected = max_gap_ratio > threshold

    return max_gap_ratio, junction_detected


def compute_intensity_statistics(gray, panel_mask, video_type):
    """
    Computes relative intensity and contrast metrics.
    For DJI_1, these are relative thermal intensity values.
    For DJI_2, these are visual intensity values.
    """

    small_gray = cv2.resize(
        gray,
        None,
        fx=0.25,
        fy=0.25,
        interpolation=cv2.INTER_AREA
    )

    small_mask = cv2.resize(
        panel_mask,
        (small_gray.shape[1], small_gray.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

    pixels = small_gray[small_mask > 0]

    if len(pixels) < 100:
        pixels = small_gray.flatten()

    pixels = pixels.astype(np.float32)

    mean_intensity = np.mean(pixels)
    median_intensity = np.median(pixels)
    max_intensity = np.max(pixels)
    min_intensity = np.min(pixels)
    std_intensity = np.std(pixels)
    dynamic_range = max_intensity - min_intensity

    p90_intensity = np.percentile(pixels, 90)
    p95_intensity = np.percentile(pixels, 95)
    p99_intensity = np.percentile(pixels, 99)

    contrast_p95 = p95_intensity - median_intensity
    contrast_max = max_intensity - median_intensity

    full_panel_pixels = gray[panel_mask > 0]

    if len(full_panel_pixels) < 100:
        full_panel_pixels = gray.flatten()

    threshold = np.percentile(full_panel_pixels, 95)

    hotspot_mask = np.zeros_like(gray, dtype=np.uint8)
    hotspot_mask[(gray >= threshold) & (panel_mask > 0)] = 255

    hot_area_pixels = np.sum(hotspot_mask > 0)
    panel_area_pixels = np.sum(panel_mask > 0)

    if panel_area_pixels > 0:
        hot_area_percentage = hot_area_pixels / panel_area_pixels
    else:
        hot_area_percentage = 0

    return {
        "mean_intensity": mean_intensity,
        "median_intensity": median_intensity,
        "max_intensity": max_intensity,
        "min_intensity": min_intensity,
        "std_intensity": std_intensity,
        "dynamic_range": dynamic_range,
        "p90_intensity": p90_intensity,
        "p95_intensity": p95_intensity,
        "p99_intensity": p99_intensity,
        "contrast_p95": contrast_p95,
        "contrast_max": contrast_max,
        "hot_area_percentage": hot_area_percentage,
        "hotspot_threshold_intensity": threshold,
        "hotspot_mask": hotspot_mask,
    }


def detect_non_pv_hot_object(panel_crop, hotspot_mask):
    """
    Detects large smooth hot or high intensity objects that are not PV panels.

    This is important for DJI_1 thermal frames where the black reference plate
    can appear as a high contrast thermal object, and for DJI_2 RGB frames where
    smooth black plates can be wrongly treated as panel surfaces.

    Logic:
    1. Find connected components in hotspot/high-intensity mask.
    2. If a large component has low edge/grid texture, mark it as non-PV object.
    """

    gray = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        hotspot_mask.astype(np.uint8),
        connectivity=8
    )

    crop_area = hotspot_mask.shape[0] * hotspot_mask.shape[1]
    cleaned_hotspot_mask = hotspot_mask.copy()

    largest_component_ratio = 0
    largest_edge_density = 0
    non_pv_hot_object_detected = False

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area <= 0:
            continue

        component_mask = labels == label
        component_ratio = area / crop_area

        component_edges = edges[component_mask]
        edge_density = np.sum(component_edges > 0) / area

        if component_ratio > largest_component_ratio:
            largest_component_ratio = component_ratio
            largest_edge_density = edge_density

        large_object = component_ratio > 0.04
        weak_grid_texture = edge_density < 0.04

        if large_object and weak_grid_texture:
            non_pv_hot_object_detected = True
            cleaned_hotspot_mask[component_mask] = 0

    return (
        non_pv_hot_object_detected,
        largest_component_ratio,
        largest_edge_density,
        cleaned_hotspot_mask
    )


def detect_smooth_non_grid_panel_object(panel_crop, panel_mask):
    """
    Detects large smooth regions that were counted as panel but do not have
    PV grid texture. This is mainly for DJI_2 RGB black plate rejection.

    A real PV module has many grid lines, cell boundaries, and texture.
    A black reference plate is usually large and smooth.
    """

    gray = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        panel_mask.astype(np.uint8),
        connectivity=8
    )

    crop_area = panel_mask.shape[0] * panel_mask.shape[1]

    largest_smooth_component_ratio = 0
    largest_smooth_edge_density = 0
    smooth_non_grid_object_detected = False

    cleaned_panel_mask = panel_mask.copy()

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area <= 0:
            continue

        component_mask = labels == label
        component_ratio = area / crop_area

        component_edges = edges[component_mask]
        edge_density = np.sum(component_edges > 0) / area

        large_component = component_ratio > 0.08
        weak_grid_texture = edge_density < 0.035

        if large_component and weak_grid_texture:
            smooth_non_grid_object_detected = True
            cleaned_panel_mask[component_mask] = 0

            if component_ratio > largest_smooth_component_ratio:
                largest_smooth_component_ratio = component_ratio
                largest_smooth_edge_density = edge_density

    return (
        smooth_non_grid_object_detected,
        largest_smooth_component_ratio,
        largest_smooth_edge_density,
        cleaned_panel_mask
    )


def hotspot_grid_location(hotspot_mask):
    """
    Divides crop into 3 x 2 grid and returns where most hotspot pixels appear.
    """

    h, w = hotspot_mask.shape
    rows = 2
    cols = 3

    max_count = 0
    best_grid = "none"

    for r in range(rows):
        for c in range(cols):
            y_start = int(r * h / rows)
            y_end = int((r + 1) * h / rows)
            x_start = int(c * w / cols)
            x_end = int((c + 1) * w / cols)

            cell = hotspot_mask[y_start:y_end, x_start:x_end]
            count = np.sum(cell > 0)

            if count > max_count:
                max_count = count
                best_grid = f"R{r + 1}C{c + 1}"

    if max_count == 0:
        best_grid = "none"

    return best_grid, max_count


def create_overlay(panel_crop, hotspot_mask):
    overlay = panel_crop.copy()
    overlay[hotspot_mask > 0] = [0, 0, 255]
    blended = cv2.addWeighted(panel_crop, 0.75, overlay, 0.25, 0)
    return blended


# ============================================================
# 4. Classification
# ============================================================

def classify_frames(df, video_name, cfg):
    video_type = cfg["video_type"]

    if video_type == "thermal":
        sharpness_threshold = cfg["sharpness_threshold"]
        panel_coverage_threshold = cfg["panel_coverage_threshold"]
        contrast_threshold = cfg["hotspot_contrast_threshold"]
        contrast_column = "contrast_p95"

    else:
        sharpness_threshold = df["laplacian_sharpness"].quantile(0.40)
        panel_coverage_threshold = df["panel_coverage"].quantile(0.30)
        contrast_threshold = df["contrast_p95"].median()
        contrast_column = "contrast_p95"

    junction_threshold = cfg["junction_threshold"]

    df["sharp_enough"] = df["laplacian_sharpness"] > sharpness_threshold
    df["panel_coverage_ok"] = df["panel_coverage"] > panel_coverage_threshold
    df["junction_ok"] = ~df["junction_detected"]
    df["non_pv_hot_object_ok"] = ~df["non_pv_hot_object_detected"]
    df["smooth_non_grid_object_ok"] = ~df["smooth_non_grid_object_detected"]

    df["usable_frame"] = (
        df["sharp_enough"] &
        df["panel_coverage_ok"] &
        df["junction_ok"] &
        df["non_pv_hot_object_ok"] &
        df["smooth_non_grid_object_ok"]
    )

    if video_type == "thermal":
        df["hotspot_detected"] = df[contrast_column] > contrast_threshold
        detection_label = "hotspot_detected"
    else:
        df["contrast_detected"] = df[contrast_column] > contrast_threshold
        detection_label = "contrast_detected"

    total_frames = len(df)
    sharp_frames = int(df["sharp_enough"].sum())
    coverage_ok_frames = int(df["panel_coverage_ok"].sum())
    junction_rejected_frames = int(df["junction_detected"].sum())
    non_pv_hot_rejected_frames = int(df["non_pv_hot_object_detected"].sum())
    smooth_non_grid_rejected_frames = int(df["smooth_non_grid_object_detected"].sum())
    usable_frames = int(df["usable_frame"].sum())
    detected_frames = int(df[detection_label].sum())

    if usable_frames > 0:
        persistence = int(df[df["usable_frame"]][detection_label].sum()) / usable_frames
        usable_df = df[df["usable_frame"]].copy()
        dominant_grid = usable_df["hotspot_grid"].mode().iloc[0]
        dominant_grid_count = int((usable_df["hotspot_grid"] == dominant_grid).sum())
        location_consistency = dominant_grid_count / usable_frames
    else:
        persistence = 0
        dominant_grid = "none"
        location_consistency = 0

    summary = {
        "video_name": video_name,
        "video_type": video_type,
        "total_sampled_frames": total_frames,

        "sharpness_threshold": sharpness_threshold,
        "panel_coverage_threshold": panel_coverage_threshold,
        "junction_threshold": junction_threshold,
        "contrast_threshold": contrast_threshold,

        "sharp_frames": sharp_frames,
        "sharp_frame_ratio": sharp_frames / total_frames if total_frames else 0,

        "coverage_ok_frames": coverage_ok_frames,
        "coverage_ok_ratio": coverage_ok_frames / total_frames if total_frames else 0,

        "junction_rejected_frames": junction_rejected_frames,
        "junction_rejected_ratio": junction_rejected_frames / total_frames if total_frames else 0,

        "non_pv_hot_object_rejected_frames": non_pv_hot_rejected_frames,
        "non_pv_hot_object_rejected_ratio": non_pv_hot_rejected_frames / total_frames if total_frames else 0,

        "smooth_non_grid_object_rejected_frames": smooth_non_grid_rejected_frames,
        "smooth_non_grid_object_rejected_ratio": smooth_non_grid_rejected_frames / total_frames if total_frames else 0,

        "usable_frames": usable_frames,
        "usable_frame_ratio": usable_frames / total_frames if total_frames else 0,

        "detected_frames": detected_frames,
        "persistence_among_usable_frames": persistence,

        "dominant_hotspot_grid": dominant_grid,
        "location_consistency": location_consistency,

        "mean_sharpness": df["laplacian_sharpness"].mean(),
        "max_sharpness": df["laplacian_sharpness"].max(),

        "mean_panel_coverage": df["panel_coverage"].mean(),
        "max_panel_coverage": df["panel_coverage"].max(),

        "mean_vertical_gap_ratio": df["max_vertical_gap_ratio"].mean(),
        "max_vertical_gap_ratio": df["max_vertical_gap_ratio"].max(),

        "mean_contrast_p95": df["contrast_p95"].mean(),
        "max_contrast_p95": df["contrast_p95"].max(),

        "mean_contrast_max": df["contrast_max"].mean(),
        "max_contrast_max": df["contrast_max"].max(),

        "mean_hot_area_percentage": df["hot_area_percentage"].mean(),
        "max_hot_area_percentage": df["hot_area_percentage"].max(),
    }

    return df, summary


# ============================================================
# 5. Main video processing
# ============================================================

def process_video(video_name, cfg):
    print("\n" + "=" * 70)
    print(f"Processing {video_name}")
    print("=" * 70)

    create_dirs(cfg)

    cap = cv2.VideoCapture(str(cfg["video_path"]))

    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {cfg['video_path']}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_video_frames / fps

    print(f"Video path: {cfg['video_path']}")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_video_frames}")
    print(f"Duration: {duration_sec:.2f} seconds")

    sample_every_n_frames = int(fps * cfg["sample_seconds"])

    x1, y1, x2, y2 = cfg["crop"]

    results = []
    frame_index = 0
    saved_index = 0

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % sample_every_n_frames == 0:
            time_sec = frame_index / fps

            frame_name = f"frame_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
            frame_path = cfg["frame_dir"] / frame_name
            cv2.imwrite(str(frame_path), frame)

            panel_crop = frame[y1:y2, x1:x2]

            crop_name = f"crop_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
            crop_path = cfg["crop_dir"] / crop_name
            cv2.imwrite(str(crop_path), panel_crop)

            if cfg["video_type"] == "thermal":
                panel_coverage, panel_mask = estimate_panel_coverage_thermal(panel_crop)
            else:
                panel_coverage, panel_mask = estimate_panel_coverage_rgb(panel_crop)

            (
                smooth_non_grid_object_detected,
                largest_smooth_component_ratio,
                largest_smooth_edge_density,
                cleaned_panel_mask
            ) = detect_smooth_non_grid_panel_object(panel_crop, panel_mask)

            panel_mask = cleaned_panel_mask
            panel_coverage = np.sum(panel_mask > 0) / panel_mask.size

            max_gap_ratio, junction_detected = detect_vertical_junction(
                panel_mask,
                cfg["junction_threshold"]
            )

            mask_name = f"mask_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
            mask_path = cfg["mask_dir"] / mask_name
            cv2.imwrite(str(mask_path), panel_mask)

            gray = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)

            lap_score = sharpness_laplacian(gray)
            sobel_score = sobel_sharpness(gray)

            stats = compute_intensity_statistics(gray, panel_mask, cfg["video_type"])
            hotspot_mask = stats.pop("hotspot_mask")

            (
                non_pv_hot_object_detected,
                largest_hot_component_ratio,
                largest_hot_component_edge_density,
                cleaned_hotspot_mask
            ) = detect_non_pv_hot_object(panel_crop, hotspot_mask)

            hotspot_mask = cleaned_hotspot_mask

            hotspot_mask_name = f"hotspot_mask_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
            hotspot_mask_path = cfg["hotspot_mask_dir"] / hotspot_mask_name
            cv2.imwrite(str(hotspot_mask_path), hotspot_mask)

            overlay = create_overlay(panel_crop, hotspot_mask)

            overlay_name = f"hotspot_overlay_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
            overlay_path = cfg["overlay_dir"] / overlay_name
            cv2.imwrite(str(overlay_path), overlay)

            hotspot_grid, hotspot_pixel_count = hotspot_grid_location(hotspot_mask)

            results.append({
                "saved_frame_id": saved_index,
                "original_frame_index": frame_index,
                "time_sec": time_sec,

                "frame_file": frame_name,
                "crop_file": crop_name,
                "panel_mask_file": mask_name,
                "hotspot_mask_file": hotspot_mask_name,
                "hotspot_overlay_file": overlay_name,

                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,

                "panel_coverage": panel_coverage,
                "max_vertical_gap_ratio": max_gap_ratio,
                "junction_detected": junction_detected,

                "smooth_non_grid_object_detected": smooth_non_grid_object_detected,
                "largest_smooth_component_ratio": largest_smooth_component_ratio,
                "largest_smooth_edge_density": largest_smooth_edge_density,

                "non_pv_hot_object_detected": non_pv_hot_object_detected,
                "largest_hot_component_ratio": largest_hot_component_ratio,
                "largest_hot_component_edge_density": largest_hot_component_edge_density,

                "laplacian_sharpness": lap_score,
                "sobel_sharpness": sobel_score,

                "mean_intensity": stats["mean_intensity"],
                "median_intensity": stats["median_intensity"],
                "max_intensity": stats["max_intensity"],
                "min_intensity": stats["min_intensity"],
                "std_intensity": stats["std_intensity"],
                "dynamic_range": stats["dynamic_range"],

                "p90_intensity": stats["p90_intensity"],
                "p95_intensity": stats["p95_intensity"],
                "p99_intensity": stats["p99_intensity"],

                "contrast_p95": stats["contrast_p95"],
                "contrast_max": stats["contrast_max"],
                "hot_area_percentage": stats["hot_area_percentage"],
                "hotspot_threshold_intensity": stats["hotspot_threshold_intensity"],

                "hotspot_grid": hotspot_grid,
                "hotspot_pixel_count": hotspot_pixel_count,
            })

            print(
                f"{video_name} frame {saved_index}: "
                f"t={time_sec:.1f}s, "
                f"coverage={panel_coverage:.2f}, "
                f"gap={max_gap_ratio:.2f}, "
                f"junction={junction_detected}, "
                f"smooth_obj={smooth_non_grid_object_detected}, "
                f"non_pv_hot={non_pv_hot_object_detected}, "
                f"sharp={lap_score:.1f}, "
                f"contrast={stats['contrast_p95']:.1f}"
            )

            saved_index += 1

        frame_index += 1

    cap.release()

    df = pd.DataFrame(results)

    if df.empty:
        raise RuntimeError(f"No frames processed for {video_name}")

    df, summary = classify_frames(df, video_name, cfg)

    analysis_csv = cfg["result_dir"] / f"{video_name}_final_rejection_analysis.csv"
    summary_csv = cfg["result_dir"] / f"{video_name}_final_rejection_summary.csv"

    df.to_csv(analysis_csv, index=False)
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    save_selected_tables(video_name, cfg, df)
    save_plots(video_name, cfg, df, summary)
    print_summary(summary)

    print(f"\nSaved analysis CSV: {analysis_csv}")
    print(f"Saved summary CSV: {summary_csv}")


# ============================================================
# 6. Save selected frame tables
# ============================================================

def save_selected_tables(video_name, cfg, df):
    result_dir = cfg["result_dir"]

    best_sharpness = df.sort_values("laplacian_sharpness", ascending=False).head(5)
    worst_sharpness = df.sort_values("laplacian_sharpness", ascending=True).head(5)

    highest_coverage = df.sort_values("panel_coverage", ascending=False).head(5)
    lowest_coverage = df.sort_values("panel_coverage", ascending=True).head(5)

    highest_gap = df.sort_values("max_vertical_gap_ratio", ascending=False).head(5)
    highest_contrast = df.sort_values("contrast_p95", ascending=False).head(5)
    largest_hot_area = df.sort_values("hot_area_percentage", ascending=False).head(5)

    rejected_non_pv_hot = df[df["non_pv_hot_object_detected"]].copy()
    rejected_smooth_non_grid = df[df["smooth_non_grid_object_detected"]].copy()

    usable_frames = df[df["usable_frame"]].copy()

    best_inspection_frames = usable_frames.sort_values(
        ["contrast_p95", "laplacian_sharpness", "panel_coverage"],
        ascending=False
    ).head(5)

    best_sharpness.to_csv(result_dir / f"{video_name}_final_best_sharpness_frames.csv", index=False)
    worst_sharpness.to_csv(result_dir / f"{video_name}_final_worst_sharpness_frames.csv", index=False)
    highest_coverage.to_csv(result_dir / f"{video_name}_final_highest_coverage_frames.csv", index=False)
    lowest_coverage.to_csv(result_dir / f"{video_name}_final_lowest_coverage_frames.csv", index=False)
    highest_gap.to_csv(result_dir / f"{video_name}_final_highest_vertical_gap_frames.csv", index=False)
    highest_contrast.to_csv(result_dir / f"{video_name}_final_highest_contrast_frames.csv", index=False)
    largest_hot_area.to_csv(result_dir / f"{video_name}_final_largest_hot_area_frames.csv", index=False)
    rejected_non_pv_hot.to_csv(result_dir / f"{video_name}_final_rejected_non_pv_hot_frames.csv", index=False)
    rejected_smooth_non_grid.to_csv(result_dir / f"{video_name}_final_rejected_smooth_non_grid_frames.csv", index=False)
    best_inspection_frames.to_csv(result_dir / f"{video_name}_final_best_inspection_frames.csv", index=False)

    if len(df) >= 3:
        random_frame = df.sample(n=1, random_state=42)
        middle_frame = df.iloc[[len(df) // 2]]
        best_single = best_inspection_frames.head(1)

        examples = pd.concat(
            [
                random_frame.assign(selection_type="Random single frame"),
                middle_frame.assign(selection_type="Middle single frame"),
                best_single.assign(selection_type="Best selected video frame"),
            ],
            ignore_index=True
        )

        examples.to_csv(result_dir / f"{video_name}_final_still_vs_video_examples.csv", index=False)


# ============================================================
# 7. Save plots
# ============================================================

def save_plots(video_name, cfg, df, summary):
    plot_dir = cfg["plot_dir"]

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_sec"], df["laplacian_sharpness"], marker="o")
    plt.axhline(
        summary["sharpness_threshold"],
        linestyle="--",
        label=f"Sharpness threshold = {summary['sharpness_threshold']:.0f}"
    )
    plt.xlabel("Time in seconds")
    plt.ylabel("Laplacian sharpness")
    plt.title(f"{video_name}: Sharpness over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{video_name}_final_sharpness_over_time.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_sec"], df["panel_coverage"] * 100, marker="o")
    plt.axhline(
        summary["panel_coverage_threshold"] * 100,
        linestyle="--",
        label=f"Coverage threshold = {summary['panel_coverage_threshold'] * 100:.1f}%"
    )
    plt.xlabel("Time in seconds")
    plt.ylabel("Panel coverage in percent")
    plt.title(f"{video_name}: Panel coverage over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{video_name}_final_panel_coverage_over_time.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_sec"], df["max_vertical_gap_ratio"] * 100, marker="o")
    plt.axhline(
        summary["junction_threshold"] * 100,
        linestyle="--",
        label=f"Junction threshold = {summary['junction_threshold'] * 100:.1f}%"
    )
    plt.xlabel("Time in seconds")
    plt.ylabel("Largest vertical gap in percent")
    plt.title(f"{video_name}: Vertical junction or obstruction over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{video_name}_final_vertical_junction_over_time.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_sec"], df["contrast_p95"], marker="o")
    plt.axhline(
        summary["contrast_threshold"],
        linestyle="--",
        label=f"Contrast threshold = {summary['contrast_threshold']:.1f}"
    )
    plt.xlabel("Time in seconds")
    plt.ylabel("P95 contrast")
    plt.title(f"{video_name}: Contrast over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{video_name}_final_contrast_over_time.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_sec"], df["hot_area_percentage"] * 100, marker="o")
    plt.xlabel("Time in seconds")
    plt.ylabel("Hot or high intensity area in percent")
    plt.title(f"{video_name}: Hot/high intensity area over time")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{video_name}_final_hot_area_percentage_over_time.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_sec"], df["non_pv_hot_object_detected"].astype(int), marker="o")
    plt.xlabel("Time in seconds")
    plt.ylabel("Non-PV hot object flag")
    plt.title(f"{video_name}: Non-PV hot object rejection")
    plt.yticks([0, 1], ["OK", "Rejected"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{video_name}_final_non_pv_hot_object_rejection.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_sec"], df["smooth_non_grid_object_detected"].astype(int), marker="o")
    plt.xlabel("Time in seconds")
    plt.ylabel("Smooth non-grid object flag")
    plt.title(f"{video_name}: Smooth non-grid object rejection")
    plt.yticks([0, 1], ["OK", "Rejected"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{video_name}_final_smooth_non_grid_rejection.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(df["time_sec"], df["usable_frame"].astype(int), marker="o")
    plt.xlabel("Time in seconds")
    plt.ylabel("Usable frame flag")
    plt.title(f"{video_name}: Final inspection usable frames over time")
    plt.yticks([0, 1], ["Not usable", "Usable"])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_dir / f"{video_name}_final_usable_frames_over_time.png", dpi=300)
    plt.close()


# ============================================================
# 8. Print summary
# ============================================================

def print_summary(summary):
    print("\nFinal summary")
    print(f"Video: {summary['video_name']}")
    print(f"Type: {summary['video_type']}")
    print(f"Total sampled frames: {summary['total_sampled_frames']}")

    print(f"Sharp frames: {summary['sharp_frames']}")
    print(f"Sharp frame ratio: {summary['sharp_frame_ratio'] * 100:.1f}%")

    print(f"Coverage OK frames: {summary['coverage_ok_frames']}")
    print(f"Coverage OK ratio: {summary['coverage_ok_ratio'] * 100:.1f}%")

    print(f"Junction rejected frames: {summary['junction_rejected_frames']}")
    print(f"Junction rejected ratio: {summary['junction_rejected_ratio'] * 100:.1f}%")

    print(f"Non-PV hot object rejected frames: {summary['non_pv_hot_object_rejected_frames']}")
    print(f"Non-PV hot object rejected ratio: {summary['non_pv_hot_object_rejected_ratio'] * 100:.1f}%")

    print(f"Smooth non-grid object rejected frames: {summary['smooth_non_grid_object_rejected_frames']}")
    print(f"Smooth non-grid object rejected ratio: {summary['smooth_non_grid_object_rejected_ratio'] * 100:.1f}%")

    print(f"Usable frames: {summary['usable_frames']}")
    print(f"Usable frame ratio: {summary['usable_frame_ratio'] * 100:.1f}%")

    print(f"Detected frames: {summary['detected_frames']}")
    print(f"Persistence among usable frames: {summary['persistence_among_usable_frames'] * 100:.1f}%")

    print(f"Dominant hotspot/grid location: {summary['dominant_hotspot_grid']}")
    print(f"Location consistency: {summary['location_consistency'] * 100:.1f}%")

    print(f"Mean panel coverage: {summary['mean_panel_coverage'] * 100:.1f}%")
    print(f"Max panel coverage: {summary['max_panel_coverage'] * 100:.1f}%")

    print(f"Mean P95 contrast: {summary['mean_contrast_p95']:.1f}")
    print(f"Max P95 contrast: {summary['max_contrast_p95']:.1f}")

    print(f"Mean hot/high area: {summary['mean_hot_area_percentage'] * 100:.2f}%")
    print(f"Max hot/high area: {summary['max_hot_area_percentage'] * 100:.2f}%")


# ============================================================
# 9. Run both videos
# ============================================================

if __name__ == "__main__":
    for video_name, cfg in VIDEOS.items():
        process_video(video_name, cfg)

    print("\nBoth DJI_1 and DJI_2 analyses completed successfully.")