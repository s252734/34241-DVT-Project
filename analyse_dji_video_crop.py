import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================
# 1. Project paths
# ============================================================

PROJECT_DIR = Path(
    r"D:\M.Sc. Autonomous Systems - DTU\Spring Semester\34241 Digital video technology\Project"
)

VIDEO_PATH = PROJECT_DIR / "DJI_1.mp4"

FRAME_DIR = PROJECT_DIR / "frames_DJI_1"
CROP_DIR = PROJECT_DIR / "cropped_frames_DJI_1"
RESULT_DIR = PROJECT_DIR / "results_DJI_1"
PLOT_DIR = PROJECT_DIR / "plots_DJI_1"

FRAME_DIR.mkdir(exist_ok=True)
CROP_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


# ============================================================
# 2. Crop coordinates for panel region
# ============================================================

# These values are selected based on the review frames.
# Format: frame[y1:y2, x1:x2]
# Adjust these later if the crop includes too much grass or misses panels.

x1, y1, x2, y2 = 0, 135, 640, 410


# ============================================================
# 3. Helper functions
# ============================================================

def sharpness_laplacian(gray):
    """
    Blur / sharpness metric.
    Higher value means sharper frame.
    Lower value means blurrier frame.
    """
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def sobel_sharpness(gray):
    """
    Gradient based sharpness metric.
    Higher value means stronger edges and better detail.
    """
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(gx ** 2 + gy ** 2))


def fast_pixel_statistics(gray):
    """
    Computes photometry / pixel statistics on cropped panel region.

    Since the video is color mapped thermal imagery, these values are
    relative pixel intensities, not calibrated temperatures.
    """

    # Downsample for faster processing
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    mean_val = np.mean(small)
    max_val = np.max(small)
    min_val = np.min(small)
    std_val = np.std(small)
    dynamic_range = max_val - min_val

    pixels = small.flatten().astype(np.float32)

    # Hottest 5 percent pixels are treated as hotspot candidate
    hot_threshold = np.percentile(pixels, 95)
    hotspot_pixels = pixels[pixels >= hot_threshold]

    hotspot_mean = np.mean(hotspot_pixels)

    # Median is used as normal background estimate
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
    Adds simple usability and detection labels.

    Thresholds are initial values and can be adjusted after checking plots.
    """

    # Sharpness threshold for usable frames
    sharpness_threshold = 800

    # Hotspot contrast threshold for detected anomaly
    hotspot_contrast_threshold = 46

    df["usable_frame"] = df["laplacian_sharpness"] > sharpness_threshold
    df["hotspot_detected"] = df["hotspot_background_contrast"] > hotspot_contrast_threshold

    total_frames = len(df)
    usable_frames = int(df["usable_frame"].sum())
    detected_frames = int(df["hotspot_detected"].sum())

    if total_frames > 0:
        usable_frame_ratio = usable_frames / total_frames
    else:
        usable_frame_ratio = 0

    if usable_frames > 0:
        detection_consistency = int(
            df[df["usable_frame"]]["hotspot_detected"].sum()
        ) / usable_frames
    else:
        detection_consistency = 0

    summary = {
        "total_sampled_frames": total_frames,
        "usable_frames": usable_frames,
        "usable_frame_ratio": usable_frame_ratio,
        "hotspot_detected_frames": detected_frames,
        "detection_consistency_among_usable_frames": detection_consistency,
        "sharpness_threshold": sharpness_threshold,
        "hotspot_contrast_threshold": hotspot_contrast_threshold,
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

# Extract around 1 frame every 1.5 seconds.
# For more frames, use int(fps)
# For fewer frames, use int(fps * 2) or int(fps * 5)

sample_every_n_frames = int(fps * 1.5)

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

        # Save original sampled frame
        frame_name = f"frame_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        frame_path = FRAME_DIR / frame_name
        cv2.imwrite(str(frame_path), frame)

        # Crop panel region
        panel_crop = frame[y1:y2, x1:x2]

        crop_name = f"crop_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        crop_path = CROP_DIR / crop_name
        cv2.imwrite(str(crop_path), panel_crop)

        # Convert cropped panel region to grayscale
        gray = cv2.cvtColor(panel_crop, cv2.COLOR_BGR2GRAY)

        # Calculate metrics
        lap_score = sharpness_laplacian(gray)
        sobel_score = sobel_sharpness(gray)

        (
            mean_val,
            max_val,
            min_val,
            std_val,
            dynamic_range,
            hotspot_mean,
            background_mean,
            hotspot_contrast,
        ) = fast_pixel_statistics(gray)

        results.append({
            "saved_frame_id": saved_index,
            "original_frame_index": frame_index,
            "time_sec": time_sec,
            "frame_file": frame_name,
            "crop_file": crop_name,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
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

        print(f"Processed frame {saved_index}: time = {time_sec:.1f}s")

        saved_index += 1

    frame_index += 1

cap.release()


# ============================================================
# 7. Save results
# ============================================================

df = pd.DataFrame(results)

if df.empty:
    raise RuntimeError("No frames were processed. Check video path and sampling settings.")

df, summary = classify_frames(df)

csv_path = RESULT_DIR / "DJI_1_panel_crop_analysis.csv"
df.to_csv(csv_path, index=False)

summary_df = pd.DataFrame([summary])
summary_csv_path = RESULT_DIR / "DJI_1_panel_crop_summary.csv"
summary_df.to_csv(summary_csv_path, index=False)

print("\nSaved panel crop analysis CSV:")
print(csv_path)

print("\nSaved summary CSV:")
print(summary_csv_path)


# ============================================================
# 8. Save best, worst, and review frame tables
# ============================================================

best_frames = df.sort_values("laplacian_sharpness", ascending=False).head(5)
worst_frames = df.sort_values("laplacian_sharpness", ascending=True).head(5)
highest_contrast = df.sort_values("hotspot_background_contrast", ascending=False).head(5)
lowest_contrast = df.sort_values("hotspot_background_contrast", ascending=True).head(5)

best_frames.to_csv(RESULT_DIR / "DJI_1_panel_crop_best_frames.csv", index=False)
worst_frames.to_csv(RESULT_DIR / "DJI_1_panel_crop_worst_frames.csv", index=False)
highest_contrast.to_csv(RESULT_DIR / "DJI_1_panel_crop_highest_contrast_frames.csv", index=False)
lowest_contrast.to_csv(RESULT_DIR / "DJI_1_panel_crop_lowest_contrast_frames.csv", index=False)

print("\nBest 5 cropped frames by sharpness:")
print(best_frames[["saved_frame_id", "time_sec", "crop_file", "laplacian_sharpness"]])

print("\nWorst 5 cropped frames by sharpness:")
print(worst_frames[["saved_frame_id", "time_sec", "crop_file", "laplacian_sharpness"]])

print("\nHighest 5 cropped frames by hotspot contrast:")
print(highest_contrast[["saved_frame_id", "time_sec", "crop_file", "hotspot_background_contrast"]])

print("\nLowest 5 cropped frames by hotspot contrast:")
print(lowest_contrast[["saved_frame_id", "time_sec", "crop_file", "hotspot_background_contrast"]])


# ============================================================
# 9. Save plots
# ============================================================

plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["laplacian_sharpness"], marker="o")
plt.axhline(
    summary["sharpness_threshold"],
    linestyle="--",
    label=f"Usable threshold = {summary['sharpness_threshold']}"
)
plt.xlabel("Time in seconds")
plt.ylabel("Laplacian sharpness score")
plt.title("Panel crop blur and sharpness variation over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "panel_crop_sharpness_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["mean_pixel"], marker="o", label="Mean pixel")
plt.plot(df["time_sec"], df["max_pixel"], marker="o", label="Max pixel")
plt.plot(df["time_sec"], df["std_pixel"], marker="o", label="Std pixel")
plt.xlabel("Time in seconds")
plt.ylabel("Relative pixel value")
plt.title("Panel crop pixel statistics over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "panel_crop_pixel_statistics_over_time.png", dpi=300)
plt.close()


plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["hotspot_background_contrast"], marker="o")
plt.axhline(
    summary["hotspot_contrast_threshold"],
    linestyle="--",
    label=f"Detection threshold = {summary['hotspot_contrast_threshold']}"
)
plt.xlabel("Time in seconds")
plt.ylabel("Hotspot to background contrast")
plt.title("Panel crop hotspot to background contrast over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "panel_crop_hotspot_contrast_over_time.png", dpi=300)
plt.close()


# ============================================================
# 10. Print final summary
# ============================================================

print("\nFinal summary:")
print(f"Total sampled frames: {summary['total_sampled_frames']}")
print(f"Usable frames: {summary['usable_frames']}")
print(f"Usable frame ratio: {summary['usable_frame_ratio'] * 100:.1f}%")
print(f"Hotspot detected frames: {summary['hotspot_detected_frames']}")
print(
    "Detection consistency among usable frames: "
    f"{summary['detection_consistency_among_usable_frames'] * 100:.1f}%"
)

print("\nSaved plots:")
print(PLOT_DIR / "panel_crop_sharpness_over_time.png")
print(PLOT_DIR / "panel_crop_pixel_statistics_over_time.png")
print(PLOT_DIR / "panel_crop_hotspot_contrast_over_time.png")

print("\nAnalysis completed successfully.")