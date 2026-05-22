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
# 2. Crop coordinates
# ============================================================

# Current panel crop region
# Format: frame[y1:y2, x1:x2]

x1, y1, x2, y2 = 0, 135, 640, 410


# ============================================================
# 3. Threshold settings
# ============================================================

SHARPNESS_THRESHOLD = 800

# This is based on your current cropped contrast distribution
HOTSPOT_CONTRAST_THRESHOLD = 46

# Minimum amount of crop that should be panel area
# Increase to 0.75 if you want stricter rejection
PANEL_COVERAGE_THRESHOLD = 0.70


# ============================================================
# 4. Helper functions
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
    Estimate how much of the crop is likely to be solar panel area.

    The DJI thermal video is color mapped.
    The panel area is mostly orange/yellow and relatively bright.
    The grass/background is mostly purple/dark/noisy.

    This mask is a practical approximation, not a perfect segmentation.
    """

    hsv = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2HSV)

    # Orange/yellow thermal panel range
    # H range may need tuning depending on color map
    lower_panel = np.array([5, 70, 80])
    upper_panel = np.array([45, 255, 255])

    panel_mask = cv2.inRange(hsv, lower_panel, upper_panel)

    # Clean small noise in mask
    kernel = np.ones((5, 5), np.uint8)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_OPEN, kernel)
    panel_mask = cv2.morphologyEx(panel_mask, cv2.MORPH_CLOSE, kernel)

    panel_coverage = np.sum(panel_mask > 0) / panel_mask.size

    return panel_coverage, panel_mask


def fast_pixel_statistics(gray, panel_mask=None):
    """
    Computes pixel statistics.

    If panel_mask is given, statistics are computed mainly on panel pixels.
    If mask has too few pixels, it falls back to the full crop.
    """

    small_gray = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    if panel_mask is not None:
        small_mask = cv2.resize(panel_mask, (small_gray.shape[1], small_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        pixels = small_gray[small_mask > 0]
    else:
        pixels = small_gray.flatten()

    if len(pixels) < 100:
        pixels = small_gray.flatten()

    pixels = pixels.astype(np.float32)

    mean_val = np.mean(pixels)
    max_val = np.max(pixels)
    min_val = np.min(pixels)
    std_val = np.std(pixels)
    dynamic_range = max_val - min_val

    hot_threshold = np.percentile(pixels, 95)
    hotspot_pixels = pixels[pixels >= hot_threshold]

    hotspot_mean = np.mean(hotspot_pixels)
    background_mean = np.median(pixels)
    hotspot_contrast = hotspot_mean - background_mean

    return (
        mean_val,
        max_val,
        min_val,
        std_val,
        dynamic_range,
        hotspot_mean,
        background_mean,
        hotspot_contrast,
    )


def classify_frames(df):
    """
    Classify frames using sharpness, panel coverage, and hotspot contrast.

    A frame is inspection usable only if:
    1. it is sharp enough
    2. it has enough visible panel area
    """

    df["sharp_enough"] = df["laplacian_sharpness"] > SHARPNESS_THRESHOLD
    df["panel_coverage_ok"] = df["panel_coverage"] > PANEL_COVERAGE_THRESHOLD

    df["usable_frame"] = (
        df["sharp_enough"] &
        df["panel_coverage_ok"]
    )

    df["hotspot_detected"] = df["hotspot_background_contrast"] > HOTSPOT_CONTRAST_THRESHOLD

    total_frames = len(df)
    sharp_frames = int(df["sharp_enough"].sum())
    coverage_ok_frames = int(df["panel_coverage_ok"].sum())
    usable_frames = int(df["usable_frame"].sum())
    hotspot_detected_frames = int(df["hotspot_detected"].sum())

    if total_frames > 0:
        sharp_frame_ratio = sharp_frames / total_frames
        coverage_ok_ratio = coverage_ok_frames / total_frames
        usable_frame_ratio = usable_frames / total_frames
    else:
        sharp_frame_ratio = 0
        coverage_ok_ratio = 0
        usable_frame_ratio = 0

    if usable_frames > 0:
        detection_consistency = int(
            df[df["usable_frame"]]["hotspot_detected"].sum()
        ) / usable_frames
    else:
        detection_consistency = 0

    summary = {
        "video_name": VIDEO_NAME,
        "total_sampled_frames": total_frames,
        "sharpness_threshold": SHARPNESS_THRESHOLD,
        "panel_coverage_threshold": PANEL_COVERAGE_THRESHOLD,
        "hotspot_contrast_threshold": HOTSPOT_CONTRAST_THRESHOLD,
        "sharp_frames": sharp_frames,
        "sharp_frame_ratio": sharp_frame_ratio,
        "coverage_ok_frames": coverage_ok_frames,
        "coverage_ok_ratio": coverage_ok_ratio,
        "usable_frames": usable_frames,
        "usable_frame_ratio": usable_frame_ratio,
        "hotspot_detected_frames": hotspot_detected_frames,
        "detection_consistency_among_usable_frames": detection_consistency,
        "mean_sharpness": df["laplacian_sharpness"].mean(),
        "max_sharpness": df["laplacian_sharpness"].max(),
        "mean_panel_coverage": df["panel_coverage"].mean(),
        "max_panel_coverage": df["panel_coverage"].max(),
        "mean_hotspot_contrast": df["hotspot_background_contrast"].mean(),
        "max_hotspot_contrast": df["hotspot_background_contrast"].max(),
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
        panel_coverage, panel_mask = estimate_panel_coverage(panel_crop)

        mask_name = f"mask_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        mask_path = MASK_DIR / mask_name
        cv2.imwrite(str(mask_path), panel_mask)

        # Convert crop to grayscale for metrics
        gray = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)

        # Sharpness metrics
        lap_score = sharpness_laplacian(gray)
        sobel_score = sobel_sharpness(gray)

        # Pixel statistics calculated mainly on panel pixels
        (
            mean_val,
            max_val,
            min_val,
            std_val,
            dynamic_range,
            hotspot_mean,
            background_mean,
            hotspot_contrast,
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
            "laplacian_sharpness": lap_score,
            "sobel_sharpness": sobel_score,
            "mean_pixel": mean_val,
            "max_pixel": max_val,
            "min_pixel": min_val,
            "std_pixel": std_val,
            "dynamic_range": dynamic_range,
            "hotspot_mean": hotspot_mean,
            "background_mean": background_mean,
            "hotspot_background_contrast": hotspot_contrast,
        })

        print(
            f"Processed frame {saved_index}: "
            f"time = {time_sec:.1f}s, "
            f"coverage = {panel_coverage:.2f}, "
            f"sharpness = {lap_score:.1f}, "
            f"contrast = {hotspot_contrast:.1f}"
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

analysis_csv_path = RESULT_DIR / f"{VIDEO_NAME}_panel_crop_coverage_analysis.csv"
summary_csv_path = RESULT_DIR / f"{VIDEO_NAME}_panel_crop_coverage_summary.csv"

df.to_csv(analysis_csv_path, index=False)
pd.DataFrame([summary]).to_csv(summary_csv_path, index=False)

print("\nSaved analysis CSV:")
print(analysis_csv_path)

print("\nSaved summary CSV:")
print(summary_csv_path)


# ============================================================
# 9. Save selected frame CSVs
# ============================================================

best_sharpness = df.sort_values("laplacian_sharpness", ascending=False).head(5)
worst_sharpness = df.sort_values("laplacian_sharpness", ascending=True).head(5)

highest_coverage = df.sort_values("panel_coverage", ascending=False).head(5)
lowest_coverage = df.sort_values("panel_coverage", ascending=True).head(5)

highest_contrast = df.sort_values("hotspot_background_contrast", ascending=False).head(5)
lowest_contrast = df.sort_values("hotspot_background_contrast", ascending=True).head(5)

usable_frames = df[df["usable_frame"]].copy()
best_inspection_frames = usable_frames.sort_values(
    ["hotspot_background_contrast", "laplacian_sharpness", "panel_coverage"],
    ascending=False
).head(5)

best_sharpness.to_csv(RESULT_DIR / f"{VIDEO_NAME}_best_sharpness_frames.csv", index=False)
worst_sharpness.to_csv(RESULT_DIR / f"{VIDEO_NAME}_worst_sharpness_frames.csv", index=False)
highest_coverage.to_csv(RESULT_DIR / f"{VIDEO_NAME}_highest_coverage_frames.csv", index=False)
lowest_coverage.to_csv(RESULT_DIR / f"{VIDEO_NAME}_lowest_coverage_frames.csv", index=False)
highest_contrast.to_csv(RESULT_DIR / f"{VIDEO_NAME}_highest_contrast_frames.csv", index=False)
lowest_contrast.to_csv(RESULT_DIR / f"{VIDEO_NAME}_lowest_contrast_frames.csv", index=False)
best_inspection_frames.to_csv(RESULT_DIR / f"{VIDEO_NAME}_best_inspection_frames.csv", index=False)

print("\nBest inspection frames:")
if best_inspection_frames.empty:
    print("No frames passed both sharpness and panel coverage thresholds.")
else:
    print(
        best_inspection_frames[
            [
                "saved_frame_id",
                "time_sec",
                "crop_file",
                "laplacian_sharpness",
                "panel_coverage",
                "hotspot_background_contrast",
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
plt.title("Panel crop sharpness over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_coverage_sharpness_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["panel_coverage"] * 100, marker="o")
plt.axhline(
    PANEL_COVERAGE_THRESHOLD * 100,
    linestyle="--",
    label=f"Coverage threshold = {PANEL_COVERAGE_THRESHOLD * 100:.0f}%"
)
plt.xlabel("Time in seconds")
plt.ylabel("Panel coverage in percent")
plt.title("Estimated panel coverage over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_panel_coverage_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["mean_pixel"], marker="o", label="Mean pixel")
plt.plot(df["time_sec"], df["max_pixel"], marker="o", label="Max pixel")
plt.plot(df["time_sec"], df["std_pixel"], marker="o", label="Std pixel")
plt.xlabel("Time in seconds")
plt.ylabel("Relative pixel value")
plt.title("Panel pixel statistics over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_coverage_pixel_statistics_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["hotspot_background_contrast"], marker="o")
plt.axhline(
    HOTSPOT_CONTRAST_THRESHOLD,
    linestyle="--",
    label=f"Detection threshold = {HOTSPOT_CONTRAST_THRESHOLD}"
)
plt.xlabel("Time in seconds")
plt.ylabel("Hotspot to background contrast")
plt.title("Panel hotspot to background contrast over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_coverage_hotspot_contrast_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["usable_frame"].astype(int), marker="o")
plt.xlabel("Time in seconds")
plt.ylabel("Usable frame flag")
plt.title("Inspection usable frames over time")
plt.yticks([0, 1], ["Not usable", "Usable"])
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / f"{VIDEO_NAME}_usable_frames_over_time.png", dpi=300)
plt.close()


# ============================================================
# 11. Print summary
# ============================================================

print("\nFinal summary")
print(f"Video: {summary['video_name']}")
print(f"Total sampled frames: {summary['total_sampled_frames']}")
print(f"Sharp frames: {summary['sharp_frames']}")
print(f"Sharp frame ratio: {summary['sharp_frame_ratio'] * 100:.1f}%")
print(f"Coverage OK frames: {summary['coverage_ok_frames']}")
print(f"Coverage OK ratio: {summary['coverage_ok_ratio'] * 100:.1f}%")
print(f"Usable frames: {summary['usable_frames']}")
print(f"Usable frame ratio: {summary['usable_frame_ratio'] * 100:.1f}%")
print(f"Hotspot detected frames: {summary['hotspot_detected_frames']}")
print(
    "Detection consistency among usable frames: "
    f"{summary['detection_consistency_among_usable_frames'] * 100:.1f}%"
)
print(f"Mean panel coverage: {summary['mean_panel_coverage'] * 100:.1f}%")
print(f"Mean hotspot contrast: {summary['mean_hotspot_contrast']:.1f}")

print("\nAnalysis completed successfully.")