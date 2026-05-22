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

FRAME_DIR = PROJECT_DIR / f"frames_{VIDEO_NAME}"
CROP_DIR = PROJECT_DIR / f"cropped_frames_{VIDEO_NAME}"
MASK_DIR = PROJECT_DIR / f"panel_masks_{VIDEO_NAME}"
RESULT_DIR = PROJECT_DIR / f"results_{VIDEO_NAME}"
PLOT_DIR = PROJECT_DIR / f"plots_{VIDEO_NAME}"

FRAME_DIR.mkdir(exist_ok=True)
CROP_DIR.mkdir(exist_ok=True)
MASK_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Crop coordinates for DJI_2
# ============================================================

# DJI_2 is RGB and has larger frame dimensions than DJI_1.
# You found that y1 = 690 and y2 = 1800 works best.
# If frame height is less than 1800, Python automatically crops to the bottom.

x1, y1, x2, y2 = 0, 690, 2048, 1800


# ============================================================
# 3. Helper functions
# ============================================================

def sharpness_laplacian(gray):
    """
    Blur and sharpness metric.
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


def estimate_panel_coverage(panel_crop):
    """
    Estimate panel coverage for RGB video.

    DJI_2 is RGB, not thermal.
    Solar panels are dark grey/blue.
    Grass is green and should be excluded.
    Bright glare and metal frames may not be counted as panel area.
    """

    hsv = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]
    s = hsv[:, :, 1]
    v = hsv[:, :, 2]

    # Green grass/background mask
    green_mask = (
        (h >= 35) & (h <= 90) &
        (s >= 35) &
        (v >= 50)
    )

    # Dark panel-like regions
    # Panels are generally dark and not green.
    dark_panel_mask = (
        (v <= 165) &
        (~green_mask)
    )

    panel_mask = dark_panel_mask.astype(np.uint8) * 255

    # Clean small noise in mask
    kernel = np.ones((7, 7), np.uint8)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_OPEN, kernel)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, kernel)

    panel_coverage = np.sum(panel_mask > 0) / panel_mask.size

    return panel_coverage, panel_mask


def detect_vertical_junction(panel_mask):
    """
    Detect large vertical non-panel gaps inside the crop.

    This rejects frames where the crop contains junctions between solar panel arrays,
    support gaps, grass gaps, or large structural interruptions.

    Returns:
    max_gap_ratio: largest continuous non-panel vertical gap divided by crop width
    junction_detected: True if the gap is too large
    """

    h, w = panel_mask.shape

    binary = panel_mask > 0

    # For each column, calculate the fraction of panel pixels
    column_panel_fraction = np.mean(binary, axis=0)

    # Column is non-panel if less than 25 percent of that column is panel
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

    # Threshold can be tuned.
    # 0.10 means reject if one continuous non-panel vertical gap is > 10% of crop width.
    junction_detected = max_gap_ratio > 0.10

    return max_gap_ratio, junction_detected


def fast_pixel_statistics(gray, panel_mask=None):
    """
    Computes pixel statistics.

    Since DJI_2 is RGB video, these are visual intensity statistics,
    not thermal temperature statistics.
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
        pixels = small_gray.flatten()

    # Fallback if panel mask fails
    if len(pixels) < 100:
        pixels = small_gray.flatten()

    pixels = pixels.astype(np.float32)

    mean_val = np.mean(pixels)
    max_val = np.max(pixels)
    min_val = np.min(pixels)
    std_val = np.std(pixels)
    dynamic_range = max_val - min_val

    # Brightest 5 percent pixels used as high contrast candidate
    high_threshold = np.percentile(pixels, 95)
    high_pixels = pixels[pixels >= high_threshold]

    high_mean = np.mean(high_pixels)
    background_mean = np.median(pixels)

    visual_contrast = high_mean - background_mean

    return (
        mean_val,
        max_val,
        min_val,
        std_val,
        dynamic_range,
        high_mean,
        background_mean,
        visual_contrast,
    )


def classify_frames(df):
    """
    Classify frames using adaptive thresholds for DJI_2.

    DJI_2 is RGB, so contrast is visual contrast, not thermal hotspot contrast.

    A frame is inspection usable only if:
    1. it is sharp enough
    2. it has enough panel coverage
    3. it does not contain a large vertical junction
    """

    sharpness_threshold = df["laplacian_sharpness"].quantile(0.40)
    panel_coverage_threshold = df["panel_coverage"].quantile(0.30)
    contrast_threshold = df["visual_contrast"].median()

    df["sharp_enough"] = df["laplacian_sharpness"] > sharpness_threshold
    df["panel_coverage_ok"] = df["panel_coverage"] > panel_coverage_threshold
    df["junction_ok"] = ~df["junction_detected"]

    df["usable_frame"] = (
        df["sharp_enough"] &
        df["panel_coverage_ok"] &
        df["junction_ok"]
    )

    df["contrast_detected"] = df["visual_contrast"] > contrast_threshold

    total_frames = len(df)
    sharp_frames = int(df["sharp_enough"].sum())
    coverage_ok_frames = int(df["panel_coverage_ok"].sum())
    junction_rejected_frames = int(df["junction_detected"].sum())
    usable_frames = int(df["usable_frame"].sum())
    contrast_detected_frames = int(df["contrast_detected"].sum())

    sharp_frame_ratio = sharp_frames / total_frames if total_frames > 0 else 0
    coverage_ok_ratio = coverage_ok_frames / total_frames if total_frames > 0 else 0
    junction_rejected_ratio = junction_rejected_frames / total_frames if total_frames > 0 else 0
    usable_frame_ratio = usable_frames / total_frames if total_frames > 0 else 0

    if usable_frames > 0:
        contrast_consistency = int(
            df[df["usable_frame"]]["contrast_detected"].sum()
        ) / usable_frames
    else:
        contrast_consistency = 0

    summary = {
        "video_name": VIDEO_NAME,
        "total_sampled_frames": total_frames,

        "sharpness_threshold": sharpness_threshold,
        "panel_coverage_threshold": panel_coverage_threshold,
        "contrast_threshold": contrast_threshold,
        "vertical_junction_threshold": 0.10,

        "sharp_frames": sharp_frames,
        "sharp_frame_ratio": sharp_frame_ratio,

        "coverage_ok_frames": coverage_ok_frames,
        "coverage_ok_ratio": coverage_ok_ratio,

        "junction_rejected_frames": junction_rejected_frames,
        "junction_rejected_ratio": junction_rejected_ratio,

        "usable_frames": usable_frames,
        "usable_frame_ratio": usable_frame_ratio,

        "contrast_detected_frames": contrast_detected_frames,
        "contrast_consistency_among_usable_frames": contrast_consistency,

        "mean_sharpness": df["laplacian_sharpness"].mean(),
        "max_sharpness": df["laplacian_sharpness"].max(),

        "mean_panel_coverage": df["panel_coverage"].mean(),
        "max_panel_coverage": df["panel_coverage"].max(),

        "mean_vertical_gap_ratio": df["max_vertical_gap_ratio"].mean(),
        "max_vertical_gap_ratio": df["max_vertical_gap_ratio"].max(),

        "mean_visual_contrast": df["visual_contrast"].mean(),
        "max_visual_contrast": df["visual_contrast"].max(),

        "mean_pixel": df["mean_pixel"].mean(),
        "max_pixel": df["max_pixel"].max(),
        "mean_std_pixel": df["std_pixel"].mean(),
    }

    return df, summary


# ============================================================
# 4. Open video
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
# 5. Frame sampling
# ============================================================

# DJI_2 is larger, so sample around 1 frame every 2 seconds.
sample_every_n_frames = int(fps * 2)

results = []
frame_index = 0
saved_index = 0


# ============================================================
# 6. Process video frames
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

        # Estimate panel mask and coverage
        panel_coverage, panel_mask = estimate_panel_coverage(panel_crop)

        # Detect vertical junctions / gaps
        max_gap_ratio, junction_detected = detect_vertical_junction(panel_mask)

        mask_name = f"mask_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        mask_path = MASK_DIR / mask_name
        cv2.imwrite(str(mask_path), panel_mask)

        # Convert crop to grayscale
        gray = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)

        # Sharpness metrics
        lap_score = sharpness_laplacian(gray)
        sobel_score = sobel_sharpness(gray)

        # Pixel and contrast statistics
        (
            mean_val,
            max_val,
            min_val,
            std_val,
            dynamic_range,
            high_mean,
            background_mean,
            visual_contrast,
        ) = fast_pixel_statistics(gray, panel_mask=panel_mask)

        results.append({
            "saved_frame_id": saved_index,
            "original_frame_index": frame_index,
            "time_sec": time_sec,

            "frame_file": frame_name,
            "crop_file": crop_name,
            "mask_file": mask_name,

            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,

            "panel_coverage": panel_coverage,
            "max_vertical_gap_ratio": max_gap_ratio,
            "junction_detected": junction_detected,

            "laplacian_sharpness": lap_score,
            "sobel_sharpness": sobel_score,

            "mean_pixel": mean_val,
            "max_pixel": max_val,
            "min_pixel": min_val,
            "std_pixel": std_val,
            "dynamic_range": dynamic_range,

            "high_intensity_mean": high_mean,
            "background_mean": background_mean,
            "visual_contrast": visual_contrast,
        })

        print(
            f"Processed frame {saved_index}: "
            f"time = {time_sec:.1f}s, "
            f"coverage = {panel_coverage:.2f}, "
            f"gap = {max_gap_ratio:.2f}, "
            f"junction = {junction_detected}, "
            f"sharpness = {lap_score:.1f}, "
            f"contrast = {visual_contrast:.1f}"
        )

        saved_index += 1

    frame_index += 1

cap.release()


# ============================================================
# 7. Save CSV results
# ============================================================

df = pd.DataFrame(results)

if df.empty:
    raise RuntimeError("No frames were processed. Check video path and sampling settings.")

df, summary = classify_frames(df)

analysis_csv_path = RESULT_DIR / f"{VIDEO_NAME}_panel_crop_coverage_analysis.csv"
summary_csv_path = RESULT_DIR / f"{VIDEO_NAME}_panel_crop_coverage_summary.csv"

df.to_csv(analysis_csv_path, index=False)
pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)

print("\nSaved analysis CSV:")
print(analysis_csv_path)

print("\nSaved summary CSV:")
print(summary_csv_path)


# ============================================================
# 8. Save selected frame CSVs
# ============================================================

best_sharpness = df.sort_values("laplacian_sharpness", ascending=False).head(5)
worst_sharpness = df.sort_values("laplacian_sharpness", ascending=True).head(5)

highest_coverage = df.sort_values("panel_coverage", ascending=False).head(5)
lowest_coverage = df.sort_values("panel_coverage", ascending=True).head(5)

highest_gap = df.sort_values("max_vertical_gap_ratio", ascending=False).head(5)
lowest_gap = df.sort_values("max_vertical_gap_ratio", ascending=True).head(5)

highest_contrast = df.sort_values("visual_contrast", ascending=False).head(5)
lowest_contrast = df.sort_values("visual_contrast", ascending=True).head(5)

usable_frames = df[df["usable_frame"]].copy()

best_inspection_frames = usable_frames.sort_values(
    ["visual_contrast", "laplacian_sharpness", "panel_coverage"],
    ascending=False
).head(5)

best_sharpness.to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_best_sharpness_frames.csv",
    index=False
)

worst_sharpness.to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_worst_sharpness_frames.csv",
    index=False
)

highest_coverage.to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_highest_coverage_frames.csv",
    index=False
)

lowest_coverage.to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_lowest_coverage_frames.csv",
    index=False
)

highest_gap.to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_highest_vertical_gap_frames.csv",
    index=False
)

lowest_gap.to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_lowest_vertical_gap_frames.csv",
    index=False
)

highest_contrast.to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_highest_visual_contrast_frames.csv",
    index=False
)

lowest_contrast.to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_lowest_visual_contrast_frames.csv",
    index=False
)

best_inspection_frames.to_csv(
    RESULT_DIR / f"{VIDEO_NAME}_best_inspection_frames.csv",
    index=False
)

print("\nBest inspection frames:")
if best_inspection_frames.empty:
    print("No frames passed sharpness, coverage, and junction filters.")
else:
    print(
        best_inspection_frames[
            [
                "saved_frame_id",
                "time_sec",
                "crop_file",
                "laplacian_sharpness",
                "panel_coverage",
                "max_vertical_gap_ratio",
                "visual_contrast",
            ]
        ]
    )


# ============================================================
# 9. Save plots
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["laplacian_sharpness"], marker="o")
plt.axhline(
    summary["sharpness_threshold"],
    linestyle="--",
    label=f"Adaptive sharpness threshold = {summary['sharpness_threshold']:.0f}"
)
plt.xlabel("Time in seconds")
plt.ylabel("Laplacian sharpness score")
plt.title(f"{VIDEO_NAME}: Panel crop sharpness over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_sharpness_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["panel_coverage"] * 100, marker="o")
plt.axhline(
    summary["panel_coverage_threshold"] * 100,
    linestyle="--",
    label=f"Adaptive coverage threshold = {summary['panel_coverage_threshold'] * 100:.1f}%"
)
plt.xlabel("Time in seconds")
plt.ylabel("Panel coverage in percent")
plt.title(f"{VIDEO_NAME}: Estimated panel coverage over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_panel_coverage_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["max_vertical_gap_ratio"] * 100, marker="o")
plt.axhline(
    summary["vertical_junction_threshold"] * 100,
    linestyle="--",
    label=f"Junction rejection threshold = {summary['vertical_junction_threshold'] * 100:.0f}%"
)
plt.xlabel("Time in seconds")
plt.ylabel("Largest vertical non-panel gap in percent")
plt.title(f"{VIDEO_NAME}: Vertical junction / obstruction over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_vertical_junction_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["mean_pixel"], marker="o", label="Mean pixel")
plt.plot(df["time_sec"], df["max_pixel"], marker="o", label="Max pixel")
plt.plot(df["time_sec"], df["std_pixel"], marker="o", label="Std pixel")
plt.xlabel("Time in seconds")
plt.ylabel("Relative pixel value")
plt.title(f"{VIDEO_NAME}: Panel pixel statistics over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_pixel_statistics_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["visual_contrast"], marker="o")
plt.axhline(
    summary["contrast_threshold"],
    linestyle="--",
    label=f"Adaptive visual contrast threshold = {summary['contrast_threshold']:.1f}"
)
plt.xlabel("Time in seconds")
plt.ylabel("Visual contrast")
plt.title(f"{VIDEO_NAME}: Visual contrast over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_visual_contrast_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["usable_frame"].astype(int), marker="o")
plt.xlabel("Time in seconds")
plt.ylabel("Usable frame flag")
plt.title(f"{VIDEO_NAME}: Inspection usable frames over time")
plt.yticks([0, 1], ["Not usable", "Usable"])
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_usable_frames_over_time.png", dpi=300)
plt.close()


# ============================================================
# 10. Print summary
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

print(f"Contrast detected frames: {summary['contrast_detected_frames']}")
print(
    "Contrast consistency among usable frames: "
    f"{summary['contrast_consistency_among_usable_frames'] * 100:.1f}%"
)

print(f"Mean panel coverage: {summary['mean_panel_coverage'] * 100:.1f}%")
print(f"Max panel coverage: {summary['max_panel_coverage'] * 100:.1f}%")

print(f"Mean vertical gap ratio: {summary['mean_vertical_gap_ratio'] * 100:.1f}%")
print(f"Max vertical gap ratio: {summary['max_vertical_gap_ratio'] * 100:.1f}%")

print(f"Mean visual contrast: {summary['mean_visual_contrast']:.1f}")
print(f"Max visual contrast: {summary['max_visual_contrast']:.1f}")

print("\nAnalysis completed successfully.")