import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


PROJECT_DIR = Path(r"D:\M.Sc. Autonomous Systems - DTU\Spring Semester\34241 Digital video technology\Project")

VIDEO_PATH = PROJECT_DIR / "DJI_1.mp4"

FRAME_DIR = PROJECT_DIR / "frames_DJI_1"
RESULT_DIR = PROJECT_DIR / "results_DJI_1"
PLOT_DIR = PROJECT_DIR / "plots_DJI_1"

FRAME_DIR.mkdir(exist_ok=True)
RESULT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)


def sharpness_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def sobel_sharpness(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    return np.mean(np.sqrt(gx ** 2 + gy ** 2))


def fast_pixel_statistics(gray):
    small = cv2.resize(gray, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)

    mean_val = np.mean(small)
    max_val = np.max(small)
    min_val = np.min(small)
    std_val = np.std(small)
    dynamic_range = max_val - min_val

    pixels = small.flatten().astype(np.float32)

    hot_threshold = np.percentile(pixels, 95)
    hotspot_pixels = pixels[pixels >= hot_threshold]

    hotspot_mean = np.mean(hotspot_pixels)
    background_mean = np.median(pixels)
    hotspot_contrast = hotspot_mean - background_mean

    return mean_val, max_val, min_val, std_val, dynamic_range, hotspot_mean, background_mean, hotspot_contrast


cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration_sec = total_frames / fps

print("Video opened successfully")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print(f"Duration: {duration_sec:.2f} seconds")

sample_every_n_frames = int(fps * 1.5)

results = []
frame_index = 0
saved_index = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    if frame_index % sample_every_n_frames == 0:
        time_sec = frame_index / fps

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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

        frame_name = f"frame_{saved_index:04d}_t_{time_sec:.1f}s.jpg"
        frame_path = FRAME_DIR / frame_name
        cv2.imwrite(str(frame_path), frame)

        results.append({
            "saved_frame_id": saved_index,
            "original_frame_index": frame_index,
            "time_sec": time_sec,
            "frame_file": frame_name,
            "laplacian_sharpness": lap_score,
            "sobel_sharpness": sobel_score,
            "mean_pixel": mean_val,
            "max_pixel": max_val,
            "min_pixel": min_val,
            "std_pixel": std_val,
            "dynamic_range": dynamic_range,
            "hotspot_mean": hotspot_mean,
            "background_mean": background_mean,
            "hotspot_background_contrast": hotspot_contrast
        })

        print(f"Processed frame {saved_index}: time {time_sec:.1f}s")
        saved_index += 1

    frame_index += 1

cap.release()

df = pd.DataFrame(results)

print(f"Total sampled frames: {len(df)}")

csv_path = RESULT_DIR / "DJI_1_frame_analysis.csv"
df.to_csv(csv_path, index=False)

best_frames = df.sort_values("laplacian_sharpness", ascending=False).head(5)
worst_frames = df.sort_values("laplacian_sharpness", ascending=True).head(5)

best_frames.to_csv(RESULT_DIR / "DJI_1_best_frames.csv", index=False)
worst_frames.to_csv(RESULT_DIR / "DJI_1_worst_frames.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["laplacian_sharpness"], marker="o")
plt.xlabel("Time in seconds")
plt.ylabel("Laplacian sharpness score")
plt.title("Blur and sharpness variation over time")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "sharpness_over_time.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["mean_pixel"], marker="o", label="Mean pixel")
plt.plot(df["time_sec"], df["max_pixel"], marker="o", label="Max pixel")
plt.plot(df["time_sec"], df["std_pixel"], marker="o", label="Std pixel")
plt.xlabel("Time in seconds")
plt.ylabel("Pixel value")
plt.title("Pixel statistics over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "pixel_statistics_over_time.png", dpi=300)
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(df["time_sec"], df["hotspot_background_contrast"], marker="o")
plt.xlabel("Time in seconds")
plt.ylabel("Hotspot to background contrast")
plt.title("Hotspot to background contrast over time")
plt.grid(True)
plt.tight_layout()
plt.savefig(PLOT_DIR / "hotspot_contrast_over_time.png", dpi=300)
plt.close()

print(f"Saved CSV to: {csv_path}")
print(f"Saved plots to: {PLOT_DIR}")
print("Analysis completed successfully.")