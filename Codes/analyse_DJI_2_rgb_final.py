import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. Project settings
# ============================================================

PROJECT_DIR = Path(
    r"D:\M.Sc. Autonomous Systems - DTU\Spring Semester\34241 Digital video technology\Project"
)

VIDEO_NAME = "DJI_2"
VIDEO_PATH = PROJECT_DIR / f"{VIDEO_NAME}.mp4"

FRAME_DIR = PROJECT_DIR / "frames_DJI_2"
CROP_DIR = PROJECT_DIR / "cropped_frames_DJI_2"
MASK_DIR = PROJECT_DIR / "panel_masks_DJI_2"
HIGH_MASK_DIR = PROJECT_DIR / "hotspot_masks_DJI_2"
OVERLAY_DIR = PROJECT_DIR / "hotspot_overlays_DJI_2"
RESULT_DIR = PROJECT_DIR / "results_DJI_2"
PLOT_DIR = PROJECT_DIR / "plots_DJI_2"

for folder in [
    FRAME_DIR,
    CROP_DIR,
    MASK_DIR,
    HIGH_MASK_DIR,
    OVERLAY_DIR,
    RESULT_DIR,
    PLOT_DIR,
]:
    folder.mkdir(exist_ok=True)


# ============================================================
# 2. DJI_2 RGB crop and thresholds
# ============================================================

# Your working crop
x1, y1, x2, y2 = 0, 690, 2048, 1800

# DJI_2 uses adaptive sharpness, coverage, and contrast thresholds
# Junction and smooth object thresholds are fixed
JUNCTION_THRESHOLD = 0.05
SMOOTH_COMPONENT_AREA_THRESHOLD = 0.08
SMOOTH_EDGE_DENSITY_THRESHOLD = 0.035


# ============================================================
# 3. Helper functions
# ============================================================

def sharpness_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def sobel_sharpness(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(gx ** 2 + gy ** 2))


def estimate_panel_coverage_rgb(panel_crop):
    """
    RGB panel mask.
    Panels are dark grey/blue.
    Grass is green and should be excluded.
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


def detect_smooth_non_grid_object(panel_crop, panel_mask):
    """
    Rejects large smooth non grid objects like black plates.
    Real PV panels have repeated grid/cell edge texture.
    """

    gray = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        panel_mask.astype(np.uint8),
        connectivity=8
    )

    crop_area = panel_mask.shape[0] * panel_mask.shape[1]

    cleaned_panel_mask = panel_mask.copy()

    smooth_object_detected = False
    largest_smooth_component_ratio = 0
    largest_smooth_edge_density = 0

    for label in range(1, num_labels):
        area = stats[label, cv2.CC_STAT_AREA]

        if area <= 0:
            continue

        component_mask = labels == label
        component_ratio = area / crop_area

        component_edges = edges[component_mask]
        edge_density = np.sum(component_edges > 0) / area

        large_component = component_ratio > SMOOTH_COMPONENT_AREA_THRESHOLD
        weak_grid_texture = edge_density < SMOOTH_EDGE_DENSITY_THRESHOLD

        if large_component and weak_grid_texture:
            smooth_object_detected = True
            cleaned_panel_mask[component_mask] = 0

            if component_ratio > largest_smooth_component_ratio:
                largest_smooth_component_ratio = component_ratio
                largest_smooth_edge_density = edge_density

    return (
        smooth_object_detected,
        largest_smooth_component_ratio,
        largest_smooth_edge_density,
        cleaned_panel_mask
    )


def detect_vertical_junction(panel_mask):
    """
    Detects large vertical non panel gaps.
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
    junction_detected = max_gap_ratio > JUNCTION_THRESHOLD

    return max_gap_ratio, junction_detected


def visual_statistics(gray, panel_mask):
    """
    RGB video visual intensity statistics.
    These are not thermal values.
    """

    small_gray = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
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

    visual_contrast_p95 = p95_intensity - median_intensity
    visual_contrast_max = max_intensity - median_intensity

    full_panel_pixels = gray[panel_mask > 0]

    if len(full_panel_pixels) < 100:
        full_panel_pixels = gray.flatten()

    high_threshold = np.percentile(full_panel_pixels, 95)

    high_mask = np.zeros_like(gray, dtype=np.uint8)
    high_mask[(gray >= high_threshold) & (panel_mask > 0)] = 255

    high_area_pixels = np.sum(high_mask > 0)
    panel_area_pixels = np.sum(panel_mask > 0)

    if panel_area_pixels > 0:
        high_area_percentage = high_area_pixels / panel_area_pixels
    else:
        high_area_percentage = 0

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
        "visual_contrast_p95": visual_contrast_p95,
        "visual_contrast_max": visual_contrast_max,
        "high_area_percentage": high_area_percentage,
        "high_threshold_intensity": high_threshold,
        "high_mask": high_mask,
    }


def grid_location(high_mask):
    h, w = high_mask.shape
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

            cell = high_mask[y_start:y_end, x_start:x_end]
            count = np.sum(cell > 0)

            if count > max_count:
                max_count = count
                best_grid = f"R{r + 1}C{c + 1}"

    if max_count == 0:
        best_grid = "none"

    return best_grid, max_count


def create_overlay(panel_crop, high_mask):
    overlay = panel_crop.copy()
    overlay[high_mask > 0] = [0, 0, 255]
    return cv2.addWeighted(panel_crop, 0.75, overlay, 0.25, 0)


def classify_frames(df):
    """
    Adaptive thresholds only for DJI_2 RGB.
    """

    sharpness_threshold = df["laplacian_sharpness"].quantile(0.40)
    panel_coverage_threshold = df["panel_coverage"].quantile(0.30)
    visual_contrast_threshold = df["visual_contrast_p95"].median()

    df["sharp_enough"] = df["laplacian_sharpness"] > sharpness_threshold
    df["panel_coverage_ok"] = df["panel_coverage"] > panel_coverage_threshold
    df["junction_ok"] = ~df["junction_detected"]
    df["smooth_object_ok"] = ~df["smooth_non_grid_object_detected"]

    df["usable_frame"] = (
        df["sharp_enough"] &
        df["panel_coverage_ok"] &
        df["junction_ok"] &
        df["smooth_object_ok"]
    )

    df["visual_contrast_detected"] = df["visual_contrast_p95"] > visual_contrast_threshold

    total_frames = len(df)
    usable_frames = int(df["usable_frame"].sum())

    if usable_frames > 0:
        contrast_persistence = int(
            df[df["usable_frame"]]["visual_contrast_detected"].sum()
        ) / usable_frames

        usable_df = df[df["usable_frame"]].copy()
        dominant_grid = usable_df["high_intensity_grid"].mode().iloc[0]
        dominant_grid_count = int((usable_df["high_intensity_grid"] == dominant_grid).sum())
        location_consistency = dominant_grid_count / usable_frames
    else:
        contrast_persistence = 0
        dominant_grid = "none"
        location_consistency = 0

    summary = {
        "video_name": VIDEO_NAME,
        "video_type": "rgb",
        "total_sampled_frames": total_frames,

        "sharpness_threshold": sharpness_threshold,
        "panel_coverage_threshold": panel_coverage_threshold,
        "junction_threshold": JUNCTION_THRESHOLD,
        "visual_contrast_threshold": visual_contrast_threshold,

        "sharp_frames": int(df["sharp_enough"].sum()),
        "sharp_frame_ratio": int(df["sharp_enough"].sum()) / total_frames if total_frames else 0,

        "coverage_ok_frames": int(df["panel_coverage_ok"].sum()),
        "coverage_ok_ratio": int(df["panel_coverage_ok"].sum()) / total_frames if total_frames else 0,

        "junction_rejected_frames": int(df["junction_detected"].sum()),
        "junction_rejected_ratio": int(df["junction_detected"].sum()) / total_frames if total_frames else 0,

        "smooth_non_grid_object_rejected_frames": int(df["smooth_non_grid_object_detected"].sum()),
        "smooth_non_grid_object_rejected_ratio": int(df["smooth_non_grid_object_detected"].sum()) / total_frames if total_frames else 0,

        "usable_frames": usable_frames,
        "usable_frame_ratio": usable_frames / total_frames if total_frames else 0,

        "visual_contrast_detected_frames": int(df["visual_contrast_detected"].sum()),
        "visual_contrast_persistence_among_usable_frames": contrast_persistence,

        "dominant_high_intensity_grid": dominant_grid,
        "high_intensity_location_consistency": location_consistency,

        "mean_sharpness": df["laplacian_sharpness"].mean(),
        "max_sharpness": df["laplacian_sharpness"].max(),

        "mean_panel_coverage": df["panel_coverage"].mean(),
        "max_panel_coverage": df["panel_coverage"].max(),

        "mean_visual_contrast_p95": df["visual_contrast_p95"].mean(),
        "max_visual_contrast_p95": df["visual_contrast_p95"].max(),

        "mean_high_area_percentage": df["high_area_percentage"].mean(),
        "max_high_area_percentage": df["high_area_percentage"].max(),
    }

    return df, summary


# ============================================================
# 4. Main processing
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_video_frames / fps

print("Video opened successfully")
print(f"Video: {VIDEO_PATH}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_video_frames}")
print(f"Duration: {duration_sec:.2f} seconds")

sample_every_n_frames = int(fps * 2.0)

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
        cv2.imwrite(str(FRAME_DIR / frame_name), frame)

        panel_crop = frame[y1:y2, x1:x2]

        crop_name = f"crop_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        cv2.imwrite(str(CROP_DIR / crop_name), panel_crop)

        panel_coverage, panel_mask = estimate_panel_coverage_rgb(panel_crop)

        (
            smooth_object_detected,
            largest_smooth_component_ratio,
            largest_smooth_edge_density,
            cleaned_panel_mask
        ) = detect_smooth_non_grid_object(panel_crop, panel_mask)

        panel_mask = cleaned_panel_mask
        panel_coverage = np.sum(panel_mask > 0) / panel_mask.size

        max_gap_ratio, junction_detected = detect_vertical_junction(panel_mask)

        panel_mask_name = f"mask_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        cv2.imwrite(str(MASK_DIR / panel_mask_name), panel_mask)

        gray = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)

        lap_score = sharpness_laplacian(gray)
        sobel_score = sobel_sharpness(gray)

        stats = visual_statistics(gray, panel_mask)

        high_mask = stats.pop("high_mask")

        high_mask_name = f"high_mask_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        cv2.imwrite(str(HIGH_MASK_DIR / high_mask_name), high_mask)

        overlay = create_overlay(panel_crop, high_mask)
        overlay_name = f"high_overlay_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        cv2.imwrite(str(OVERLAY_DIR / overlay_name), overlay)

        high_grid, high_pixel_count = grid_location(high_mask)

        results.append({
            "saved_frame_id": saved_index,
            "original_frame_index": frame_index,
            "time_sec": time_sec,

            "frame_file": frame_name,
            "crop_file": crop_name,
            "panel_mask_file": panel_mask_name,
            "high_mask_file": high_mask_name,
            "high_overlay_file": overlay_name,

            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,

            "panel_coverage": panel_coverage,
            "max_vertical_gap_ratio": max_gap_ratio,
            "junction_detected": junction_detected,

            "smooth_non_grid_object_detected": smooth_object_detected,
            "largest_smooth_component_ratio": largest_smooth_component_ratio,
            "largest_smooth_edge_density": largest_smooth_edge_density,

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

            "visual_contrast_p95": stats["visual_contrast_p95"],
            "visual_contrast_max": stats["visual_contrast_max"],
            "high_area_percentage": stats["high_area_percentage"],
            "high_threshold_intensity": stats["high_threshold_intensity"],

            "high_intensity_grid": high_grid,
            "high_pixel_count": high_pixel_count,
        })

        print(
            f"Frame {saved_index}: "
            f"t={time_sec:.1f}s, "
            f"coverage={panel_coverage:.2f}, "
            f"junction={junction_detected}, "
            f"smooth_obj={smooth_object_detected}, "
            f"sharpness={lap_score:.1f}, "
            f"visual contrast={stats['visual_contrast_p95']:.1f}"
        )

        saved_index += 1

    frame_index += 1

cap.release()

df = pd.DataFrame(results)

if df.empty:
    raise RuntimeError("No frames were processed.")

df, summary = classify_frames(df)

analysis_csv = RESULT_DIR / f"{VIDEO_NAME}_rgb_final_analysis.csv"
summary_csv = RESULT_DIR / f"{VIDEO_NAME}_rgb_final_summary.csv"

df.to_csv(analysis_csv, index=False)
pd.DataFrame([summary]).to_csv(summary_csv, index=False)


# ============================================================
# 5. Save selected tables
# ============================================================

best_inspection_frames = df[df["usable_frame"]].sort_values(
    ["visual_contrast_p95", "laplacian_sharpness", "panel_coverage"],
    ascending=False
).head(5)

df.sort_values("laplacian_sharpness", ascending=False).head(5).to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_rgb_final_best_sharpness_frames.csv",
    index=False
)

df.sort_values("laplacian_sharpness", ascending=True).head(5).to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_rgb_final_worst_sharpness_frames.csv",
    index=False
)

df.sort_values("visual_contrast_p95", ascending=False).head(5).to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_rgb_final_highest_visual_contrast_frames.csv",
    index=False
)

df.sort_values("max_vertical_gap_ratio", ascending=False).head(5).to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_rgb_final_highest_vertical_gap_frames.csv",
    index=False
)

df[df["smooth_non_grid_object_detected"]].to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_rgb_final_rejected_smooth_non_grid_frames.csv",
    index=False
)

best_inspection_frames.to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_rgb_final_best_inspection_frames.csv",
    index=False
)

random_frame = df.sample(n=1, random_state=42)
middle_frame = df.iloc[[len(df) // 2]]
best_single = best_inspection_frames.head(1)

still_vs_video_examples = pd.concat(
    [
        random_frame.assign(selection_type="Random single frame"),
        middle_frame.assign(selection_type="Middle single frame"),
        best_single.assign(selection_type="Best selected video frame"),
    ],
    ignore_index=True
)

still_vs_video_examples.to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_rgb_final_still_vs_video_examples.csv",
    index=False
)


# ============================================================
# 6. Save plots
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["laplacian_sharpness"], marker="o")
plt.axhline(summary["sharpness_threshold"], linestyle="--", label=f"Adaptive threshold = {summary['sharpness_threshold']:.0f}")
plt.xlabel("Time in seconds")
plt.ylabel("Laplacian sharpness")
plt.title("DJI_2 RGB sharpness over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_rgb_final_sharpness_over_time.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["panel_coverage"] * 100, marker="o")
plt.axhline(summary["panel_coverage_threshold"] * 100, linestyle="--", label=f"Adaptive threshold = {summary['panel_coverage_threshold'] * 100:.1f}%")
plt.xlabel("Time in seconds")
plt.ylabel("Panel coverage in percent")
plt.title("DJI_2 RGB panel coverage over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_rgb_final_panel_coverage_over_time.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["max_vertical_gap_ratio"] * 100, marker="o")
plt.axhline(JUNCTION_THRESHOLD * 100, linestyle="--", label=f"Junction threshold = {JUNCTION_THRESHOLD * 100:.1f}%")
plt.xlabel("Time in seconds")
plt.ylabel("Largest vertical gap in percent")
plt.title("DJI_2 RGB vertical junction over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_rgb_final_vertical_junction_over_time.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["visual_contrast_p95"], marker="o")
plt.axhline(summary["visual_contrast_threshold"], linestyle="--", label=f"Adaptive threshold = {summary['visual_contrast_threshold']:.1f}")
plt.xlabel("Time in seconds")
plt.ylabel("P95 visual contrast")
plt.title("DJI_2 RGB visual contrast over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_rgb_final_visual_contrast_over_time.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["smooth_non_grid_object_detected"].astype(int), marker="o")
plt.xlabel("Time in seconds")
plt.ylabel("Smooth object flag")
plt.title("DJI_2 smooth non grid object rejection")
plt.yticks([0, 1], ["OK", "Rejected"])
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_rgb_final_smooth_non_grid_rejection.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["usable_frame"].astype(int), marker="o")
plt.xlabel("Time in seconds")
plt.ylabel("Usable frame flag")
plt.title("DJI_2 final usable RGB frames")
plt.yticks([0, 1], ["Not usable", "Usable"])
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_rgb_final_usable_frames_over_time.png", dpi=300)
plt.close()


# ============================================================
# 7. Print summary
# ============================================================

print("\nFinal DJI_2 RGB summary")
for key, value in summary.items():
    print(f"{key}: {value}")

print("\nSaved:")
print(analysis_csv)
print(summary_csv)
print("\nDJI_2 RGB analysis completed.")