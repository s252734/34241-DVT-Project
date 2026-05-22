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

VIDEO_NAME = "DJI_1"
VIDEO_PATH = PROJECT_DIR / f"{VIDEO_NAME}.mp4"

# New output folders for improved thermal video analysis
FRAME_DIR = PROJECT_DIR / f"v2_frames_{VIDEO_NAME}"
CROP_DIR = PROJECT_DIR / f"v2_cropped_frames_{VIDEO_NAME}"
MASK_DIR = PROJECT_DIR / f"v2_panel_masks_{VIDEO_NAME}"
HOTSPOT_MASK_DIR = PROJECT_DIR / f"v2_hotspot_masks_{VIDEO_NAME}"
OVERLAY_DIR = PROJECT_DIR / f"v2_hotspot_overlays_{VIDEO_NAME}"
RESULT_DIR = PROJECT_DIR / f"v2_results_{VIDEO_NAME}"
PLOT_DIR = PROJECT_DIR / f"v2_plots_{VIDEO_NAME}"

FRAME_DIR.mkdir(exist_ok=True)
CROP_DIR.mkdir(exist_ok=True)
MASK_DIR.mkdir(exist_ok=True)
HOTSPOT_MASK_DIR.mkdir(exist_ok=True)
OVERLAY_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Crop coordinates for DJI_1 thermal video
# ============================================================

# DJI_1 thermal frame is smaller than DJI_2 RGB frame.
# These coordinates were selected from the previous working crop.
# Format: frame[y1:y2, x1:x2]

x1, y1, x2, y2 = 0, 135, 640, 410


# ============================================================
# 3. Fixed quality thresholds for DJI_1 thermal video
# ============================================================

SHARPNESS_THRESHOLD = 800
PANEL_COVERAGE_THRESHOLD = 0.70

# Hotspot threshold is now derived per frame using P95 and median.
# This constant is used only for a simple detection rule.
HOTSPOT_CONTRAST_THRESHOLD = 30

# Hot pixels are defined as pixels above the 95th percentile of the panel region.
HOT_PERCENTILE = 95


# ============================================================
# 4. Helper functions
# ============================================================

def sharpness_laplacian(gray):
    """
    Blur / sharpness metric.
    Higher value means sharper frame.
    """
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def sobel_sharpness(gray):
    """
    Gradient based sharpness metric.
    Higher value means stronger edges and clearer detail.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(gx ** 2 + gy ** 2))


def estimate_panel_coverage_thermal(panel_crop):
    """
    Estimate panel coverage for DJI_1 thermal video.

    DJI_1 is color mapped thermal imagery.
    Panel areas are mostly orange/yellow.
    Background/grass is mostly purple/dark/noisy.

    This is an approximate segmentation, not perfect physical calibration.
    """

    hsv = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2HSV)

    # Orange/yellow thermal panel range
    lower_panel = np.array([5, 70, 80])
    upper_panel = np.array([45, 255, 255])

    panel_mask = cv2.inRange(hsv, lower_panel, upper_panel)

    kernel = np.ones((5, 5), np.uint8)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_OPEN, kernel)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, kernel)

    panel_coverage = np.sum(panel_mask > 0) / panel_mask.size

    return panel_coverage, panel_mask


def detect_vertical_junction(panel_mask):
    """
    Detect large vertical non-panel gaps in the thermal crop.

    This prevents frames with array gaps or strong vertical obstructions
    from being treated as inspection usable.
    """

    h, w = panel_mask.shape
    binary = panel_mask > 0

    column_panel_fraction = np.mean(binary, axis=0)

    # Column is treated as non-panel if less than 25 percent is panel
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

    # DJI_1 thermal crop can contain support structures.
    # Reject only large continuous vertical gaps.
    junction_detected = max_gap_ratio > 0.08

    return max_gap_ratio, junction_detected


def thermal_statistics(gray, panel_mask=None):
    """
    Calculate still-image-style IRT metrics on one usable video frame.

    Since DJI_1 is color mapped thermal video, the values are relative
    intensity values, not calibrated temperature values.
    """

    small_gray = cv2.resize(
        gray,
        None,
        fx=0.25,
        fy=0.25,
        interpolation=cv2.INTER_AREA
    )

    if panel_mask is not None:
        small_mask = cv2.resize(
            panel_mask,
            (small_gray.shape[1], small_gray.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )
        pixels = small_gray[small_mask > 0]
    else:
        small_mask = None
        pixels = small_gray.flatten()

    # Fallback if panel mask fails
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
    p95_intensity = np.percentile(pixels, HOT_PERCENTILE)
    p99_intensity = np.percentile(pixels, 99)

    hotspot_contrast_p95 = p95_intensity - median_intensity
    hotspot_contrast_max = max_intensity - median_intensity

    # Hotspot mask on full crop resolution
    if panel_mask is not None:
        panel_pixels_full = gray[panel_mask > 0]
        if len(panel_pixels_full) >= 100:
            full_threshold = np.percentile(panel_pixels_full, HOT_PERCENTILE)
        else:
            full_threshold = np.percentile(gray.flatten(), HOT_PERCENTILE)
    else:
        full_threshold = np.percentile(gray.flatten(), HOT_PERCENTILE)

    hotspot_mask = np.zeros_like(gray, dtype=np.uint8)

    if panel_mask is not None:
        hotspot_mask[(gray >= full_threshold) & (panel_mask > 0)] = 255
        panel_area_pixels = np.sum(panel_mask > 0)
    else:
        hotspot_mask[gray >= full_threshold] = 255
        panel_area_pixels = gray.size

    hot_area_pixels = np.sum(hotspot_mask > 0)

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
        "hotspot_contrast_p95": hotspot_contrast_p95,
        "hotspot_contrast_max": hotspot_contrast_max,
        "hot_area_percentage": hot_area_percentage,
        "hotspot_threshold_intensity": full_threshold,
        "hotspot_mask": hotspot_mask,
    }


def hotspot_grid_location(hotspot_mask):
    """
    Divide crop into 3 columns x 2 rows and locate where most hotspot pixels occur.

    This gives a simple temporal consistency indicator:
    if the same grid cell appears repeatedly, the hotspot is more persistent.
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


def create_hotspot_overlay(panel_crop, hotspot_mask):
    """
    Create a visual overlay showing detected hotspot pixels.
    """

    overlay = panel_crop.copy()

    # Mark hotspot pixels in red
    overlay[hotspot_mask > 0] = [0, 0, 255]

    blended = cv2.addWeighted(panel_crop, 0.75, overlay, 0.25, 0)

    return blended


def classify_frames(df):
    """
    Classify frames using quality and thermal inspection metrics.

    Usable frame:
    sharp enough + panel coverage OK + no large junction

    Hotspot detected:
    P95 contrast above threshold
    """

    df["sharp_enough"] = df["laplacian_sharpness"] > SHARPNESS_THRESHOLD
    df["panel_coverage_ok"] = df["panel_coverage"] > PANEL_COVERAGE_THRESHOLD
    df["junction_ok"] = ~df["junction_detected"]

    df["usable_frame"] = (
        df["sharp_enough"] &
        df["panel_coverage_ok"] &
        df["junction_ok"]
    )

    df["hotspot_detected"] = (
        df["hotspot_contrast_p95"] > HOTSPOT_CONTRAST_THRESHOLD
    )

    total_frames = len(df)
    sharp_frames = int(df["sharp_enough"].sum())
    coverage_ok_frames = int(df["panel_coverage_ok"].sum())
    junction_rejected_frames = int(df["junction_detected"].sum())
    usable_frames = int(df["usable_frame"].sum())
    hotspot_detected_frames = int(df["hotspot_detected"].sum())

    sharp_frame_ratio = sharp_frames / total_frames if total_frames > 0 else 0
    coverage_ok_ratio = coverage_ok_frames / total_frames if total_frames > 0 else 0
    junction_rejected_ratio = junction_rejected_frames / total_frames if total_frames > 0 else 0
    usable_frame_ratio = usable_frames / total_frames if total_frames > 0 else 0

    if usable_frames > 0:
        hotspot_persistence_index = int(
            df[df["usable_frame"]]["hotspot_detected"].sum()
        ) / usable_frames
    else:
        hotspot_persistence_index = 0

    usable_df = df[df["usable_frame"]].copy()

    if len(usable_df) > 0:
        dominant_hotspot_grid = usable_df["hotspot_grid"].mode().iloc[0]
        dominant_grid_count = int((usable_df["hotspot_grid"] == dominant_hotspot_grid).sum())
        hotspot_location_consistency = dominant_grid_count / len(usable_df)
    else:
        dominant_hotspot_grid = "none"
        hotspot_location_consistency = 0

    summary = {
        "video_name": VIDEO_NAME,
        "total_sampled_frames": total_frames,

        "sharpness_threshold": SHARPNESS_THRESHOLD,
        "panel_coverage_threshold": PANEL_COVERAGE_THRESHOLD,
        "vertical_junction_threshold": 0.15,
        "hotspot_contrast_threshold": HOTSPOT_CONTRAST_THRESHOLD,

        "sharp_frames": sharp_frames,
        "sharp_frame_ratio": sharp_frame_ratio,

        "coverage_ok_frames": coverage_ok_frames,
        "coverage_ok_ratio": coverage_ok_ratio,

        "junction_rejected_frames": junction_rejected_frames,
        "junction_rejected_ratio": junction_rejected_ratio,

        "usable_frames": usable_frames,
        "usable_frame_ratio": usable_frame_ratio,

        "hotspot_detected_frames": hotspot_detected_frames,
        "hotspot_persistence_index_among_usable_frames": hotspot_persistence_index,

        "dominant_hotspot_grid": dominant_hotspot_grid,
        "hotspot_location_consistency": hotspot_location_consistency,

        "mean_sharpness": df["laplacian_sharpness"].mean(),
        "max_sharpness": df["laplacian_sharpness"].max(),

        "mean_panel_coverage": df["panel_coverage"].mean(),
        "max_panel_coverage": df["panel_coverage"].max(),

        "mean_intensity": df["mean_intensity"].mean(),
        "max_intensity": df["max_intensity"].max(),
        "mean_std_intensity": df["std_intensity"].mean(),

        "mean_hotspot_contrast_p95": df["hotspot_contrast_p95"].mean(),
        "max_hotspot_contrast_p95": df["hotspot_contrast_p95"].max(),

        "mean_hotspot_contrast_max": df["hotspot_contrast_max"].mean(),
        "max_hotspot_contrast_max": df["hotspot_contrast_max"].max(),

        "mean_hot_area_percentage": df["hot_area_percentage"].mean(),
        "max_hot_area_percentage": df["hot_area_percentage"].max(),
    }

    return df, summary


# ============================================================
# 5. Open video
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_video_frames / fps

print("Video opened successfully")
print(f"Video path: {VIDEO_PATH}")
print(f"FPS: {fps:.2f}")
print(f"Total video frames: {total_video_frames}")
print(f"Duration: {duration_sec:.2f} seconds")


# ============================================================
# 6. Frame sampling
# ============================================================

# Around 1 frame every 1.5 seconds
sample_every_n_frames = int(fps * 1.5)

results = []
frame_index = 0
saved_index = 0


# ============================================================
# 7. Process video
# ============================================================

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_index % sample_every_n_frames == 0:
        time_sec = frame_index / fps

        # Save full sampled frame
        frame_name = f"frame_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        frame_path = FRAME_DIR / frame_name
        cv2.imwrite(str(frame_path), frame)

        # Crop panel region
        panel_crop = frame[y1:y2, x1:x2]

        crop_name = f"crop_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        crop_path = CROP_DIR / crop_name
        cv2.imwrite(str(crop_path), panel_crop)

        # Estimate panel coverage
        panel_coverage, panel_mask = estimate_panel_coverage_thermal(panel_crop)

        # Detect large vertical gap / junction
        max_gap_ratio, junction_detected = detect_vertical_junction(panel_mask)

        mask_name = f"mask_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        mask_path = MASK_DIR / mask_name
        cv2.imwrite(str(mask_path), panel_mask)

        # Convert crop to grayscale intensity
        gray = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)

        # Sharpness metrics
        lap_score = sharpness_laplacian(gray)
        sobel_score = sobel_sharpness(gray)

        # Thermal relative intensity metrics
        stats = thermal_statistics(gray, panel_mask=panel_mask)

        hotspot_mask = stats.pop("hotspot_mask")

        hotspot_mask_name = f"hotspot_mask_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        hotspot_mask_path = HOTSPOT_MASK_DIR / hotspot_mask_name
        cv2.imwrite(str(hotspot_mask_path), hotspot_mask)

        overlay = create_hotspot_overlay(panel_crop, hotspot_mask)

        overlay_name = f"hotspot_overlay_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        overlay_path = OVERLAY_DIR / overlay_name
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

            "hotspot_contrast_p95": stats["hotspot_contrast_p95"],
            "hotspot_contrast_max": stats["hotspot_contrast_max"],
            "hot_area_percentage": stats["hot_area_percentage"],
            "hotspot_threshold_intensity": stats["hotspot_threshold_intensity"],

            "hotspot_grid": hotspot_grid,
            "hotspot_pixel_count": hotspot_pixel_count,
        })

        print(
            f"Processed frame {saved_index}: "
            f"time = {time_sec:.1f}s, "
            f"coverage = {panel_coverage:.2f}, "
            f"gap = {max_gap_ratio:.2f}, "
            f"junction = {junction_detected}, "
            f"sharpness = {lap_score:.1f}, "
            f"P95 contrast = {stats['hotspot_contrast_p95']:.1f}, "
            f"hot area = {stats['hot_area_percentage'] * 100:.1f}%"
        )

        saved_index += 1

    frame_index += 1

cap.release()


# ============================================================
# 8. Save CSV results
# ============================================================

df = pd.DataFrame(results)

if df.empty:
    raise RuntimeError("No frames were processed. Check video path and sampling settings.")

df, summary = classify_frames(df)

analysis_csv_path = RESULT_DIR / f"{VIDEO_NAME}_thermal_video_v2_analysis.csv"
summary_csv_path = RESULT_DIR / f"{VIDEO_NAME}_thermal_video_v2_summary.csv"

df.to_csv(analysis_csv_path, index=False)
pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)

print("\nSaved analysis CSV:")
print(analysis_csv_path)

print("\nSaved summary CSV:")
print(summary_csv_path)


# ============================================================
# 9. Save selected frame tables
# ============================================================

best_sharpness = df.sort_values("laplacian_sharpness", ascending=False).head(5)
worst_sharpness = df.sort_values("laplacian_sharpness", ascending=True).head(5)

highest_coverage = df.sort_values("panel_coverage", ascending=False).head(5)
lowest_coverage = df.sort_values("panel_coverage", ascending=True).head(5)

highest_contrast = df.sort_values("hotspot_contrast_p95", ascending=False).head(5)
lowest_contrast = df.sort_values("hotspot_contrast_p95", ascending=True).head(5)

largest_hot_area = df.sort_values("hot_area_percentage", ascending=False).head(5)

usable_frames = df[df["usable_frame"]].copy()

best_inspection_frames = usable_frames.sort_values(
    ["hotspot_contrast_p95", "laplacian_sharpness", "panel_coverage"],
    ascending=False
).head(5)

random_frame = df.sample(n=1, random_state=42)
middle_frame = df.iloc[[len(df) // 2]]
best_single_frame = best_inspection_frames.head(1)

still_vs_video_examples = pd.concat(
    [
        random_frame.assign(selection_type="Random single frame"),
        middle_frame.assign(selection_type="Middle single frame"),
        best_single_frame.assign(selection_type="Best selected video frame"),
    ],
    ignore_index=True
)

best_sharpness.to_csv(RESULT_DIR / f"{VIDEO_NAME}_v2_best_sharpness_frames.csv", index=False)
worst_sharpness.to_csv(RESULT_DIR / f"{VIDEO_NAME}_v2_worst_sharpness_frames.csv", index=False)
highest_coverage.to_csv(RESULT_DIR / f"{VIDEO_NAME}_v2_highest_coverage_frames.csv", index=False)
lowest_coverage.to_csv(RESULT_DIR / f"{VIDEO_NAME}_v2_lowest_coverage_frames.csv", index=False)
highest_contrast.to_csv(RESULT_DIR / f"{VIDEO_NAME}_v2_highest_hotspot_contrast_frames.csv", index=False)
lowest_contrast.to_csv(RESULT_DIR / f"{VIDEO_NAME}_v2_lowest_hotspot_contrast_frames.csv", index=False)
largest_hot_area.to_csv(RESULT_DIR / f"{VIDEO_NAME}_v2_largest_hot_area_frames.csv", index=False)
best_inspection_frames.to_csv(RESULT_DIR / f"{VIDEO_NAME}_v2_best_inspection_frames.csv", index=False)
still_vs_video_examples.to_csv(RESULT_DIR / f"{VIDEO_NAME}_v2_still_vs_video_examples.csv", index=False)

print("\nBest inspection frames:")
if best_inspection_frames.empty:
    print("No frames passed quality filters.")
else:
    print(
        best_inspection_frames[
            [
                "saved_frame_id",
                "time_sec",
                "crop_file",
                "laplacian_sharpness",
                "panel_coverage",
                "hotspot_contrast_p95",
                "hot_area_percentage",
                "hotspot_grid",
            ]
        ]
    )


# ============================================================
# 10. Save plots
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["laplacian_sharpness"], marker="o")
plt.axhline(
    SHARPNESS_THRESHOLD,
    linestyle="--",
    label=f"Sharpness threshold = {SHARPNESS_THRESHOLD}"
)
plt.xlabel("Time in seconds")
plt.ylabel("Laplacian sharpness score")
plt.title(f"{VIDEO_NAME}: Thermal frame sharpness over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_v2_sharpness_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["panel_coverage"] * 100, marker="o")
plt.axhline(
    PANEL_COVERAGE_THRESHOLD * 100,
    linestyle="--",
    label=f"Panel coverage threshold = {PANEL_COVERAGE_THRESHOLD * 100:.0f}%"
)
plt.xlabel("Time in seconds")
plt.ylabel("Panel coverage in percent")
plt.title(f"{VIDEO_NAME}: Panel coverage over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_v2_panel_coverage_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["max_vertical_gap_ratio"] * 100, marker="o")
plt.axhline(
    8,
    linestyle="--",
    label="Junction rejection threshold = 8%"
)
plt.xlabel("Time in seconds")
plt.ylabel("Largest vertical non-panel gap in percent")
plt.title(f"{VIDEO_NAME}: Junction / obstruction over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_v2_junction_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["mean_intensity"], marker="o", label="Mean intensity")
plt.plot(df["time_sec"], df["max_intensity"], marker="o", label="Max intensity")
plt.plot(df["time_sec"], df["std_intensity"], marker="o", label="Std intensity")
plt.xlabel("Time in seconds")
plt.ylabel("Relative thermal intensity")
plt.title(f"{VIDEO_NAME}: Relative thermal intensity statistics")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_v2_thermal_statistics_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["hotspot_contrast_p95"], marker="o")
plt.axhline(
    HOTSPOT_CONTRAST_THRESHOLD,
    linestyle="--",
    label=f"Hotspot contrast threshold = {HOTSPOT_CONTRAST_THRESHOLD}"
)
plt.xlabel("Time in seconds")
plt.ylabel("P95 hotspot contrast")
plt.title(f"{VIDEO_NAME}: Hotspot contrast over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_v2_hotspot_contrast_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["hot_area_percentage"] * 100, marker="o")
plt.xlabel("Time in seconds")
plt.ylabel("Hot area percentage")
plt.title(f"{VIDEO_NAME}: Hot area percentage over time")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_v2_hot_area_percentage_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["usable_frame"].astype(int), marker="o")
plt.xlabel("Time in seconds")
plt.ylabel("Usable frame flag")
plt.title(f"{VIDEO_NAME}: Inspection usable frames over time")
plt.yticks([0, 1], ["Not usable", "Usable"])
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_v2_usable_frames_over_time.png", dpi=300)
plt.close()


# ============================================================
# 11. Print final summary
# ============================================================

print("\nFinal summary")
print(f"Video: {summary['video_name']}")
print(f"Total sampled frames: {summary['total_sampled_frames']}")

print(f"Sharp frames: {summary['sharp_frames']}")
print(f"Sharp frame ratio: {summary['sharp_frame_ratio'] * 100:.1f}%")

print(f"Coverage OK frames: {summary['coverage_ok_frames']}")
print(f"Coverage OK ratio: {summary['coverage_ok_ratio'] * 100:.1f}%")

print(f"Junction rejected frames: {summary['junction_rejected_frames']}")
print(f"Junction rejected ratio: {summary['junction_rejected_ratio'] * 100:.1f}%")

print(f"Usable frames: {summary['usable_frames']}")
print(f"Usable frame ratio: {summary['usable_frame_ratio'] * 100:.1f}%")

print(f"Hotspot detected frames: {summary['hotspot_detected_frames']}")
print(
    "Hotspot persistence index among usable frames: "
    f"{summary['hotspot_persistence_index_among_usable_frames'] * 100:.1f}%"
)

print(f"Dominant hotspot grid: {summary['dominant_hotspot_grid']}")
print(
    "Hotspot location consistency: "
    f"{summary['hotspot_location_consistency'] * 100:.1f}%"
)

print(f"Mean panel coverage: {summary['mean_panel_coverage'] * 100:.1f}%")
print(f"Max panel coverage: {summary['max_panel_coverage'] * 100:.1f}%")

print(f"Mean P95 hotspot contrast: {summary['mean_hotspot_contrast_p95']:.1f}")
print(f"Max P95 hotspot contrast: {summary['max_hotspot_contrast_p95']:.1f}")

print(f"Mean hot area percentage: {summary['mean_hot_area_percentage'] * 100:.2f}%")
print(f"Max hot area percentage: {summary['max_hot_area_percentage'] * 100:.2f}%")

print("\nAnalysis completed successfully.")