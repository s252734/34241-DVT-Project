import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import colour
from colour.plotting import plot_chromaticity_diagram_CIE1931


# ============================================================
# 1. Project settings
# ============================================================

PROJECT_DIR = Path(
    r"D:\M.Sc. Autonomous Systems - DTU\Spring Semester\34241 Digital video technology\Project"
)

VIDEO_NAME = "DJI_2"
VIDEO_PATH = PROJECT_DIR / f"{VIDEO_NAME}.mp4"

OUTPUT_DIR = PROJECT_DIR / "plots_DJI_2"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "DJI_2_CIE1931_panel_chromaticity.png"


# ============================================================
# 2. DJI_2 crop
# ============================================================

# Your working crop for DJI_2
x1, y1, x2, y2 = 0, 690, 2048, 1800


# ============================================================
# 3. Sampling settings
# ============================================================

SAMPLE_EVERY_SECONDS = 2.0
MAX_PIXELS_PER_FRAME = 4000


# ============================================================
# 4. Helper functions
# ============================================================

def create_rgb_panel_mask(crop_bgr):
    """
    Create a panel mask for RGB frames.

    Green grass is removed.
    Dark non green regions are treated as possible panel regions.
    """

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

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

    return panel_mask


def srgb_to_xy(rgb_pixels):
    """
    Convert RGB pixels to CIE xy chromaticity coordinates.

    Input:
    rgb_pixels in range 0 to 255, shape N x 3

    Output:
    xy chromaticity coordinates, shape N x 2
    """

    rgb = rgb_pixels.astype(np.float32) / 255.0

    # Convert sRGB to XYZ using colour science
    xyz = colour.sRGB_to_XYZ(rgb)

    X = xyz[:, 0]
    Y = xyz[:, 1]
    Z = xyz[:, 2]

    denom = X + Y + Z

    valid = denom > 1e-8

    x = np.zeros_like(X)
    y = np.zeros_like(Y)

    x[valid] = X[valid] / denom[valid]
    y[valid] = Y[valid] / denom[valid]

    xy = np.column_stack([x[valid], y[valid]])

    return xy


# ============================================================
# 5. Open video and collect panel RGB pixels
# ============================================================

cap = cv2.VideoCapture(str(VIDEO_PATH))

if not cap.isOpened():
    raise FileNotFoundError(f"Could not open video: {VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = total_frames / fps

sample_every_n_frames = int(fps * SAMPLE_EVERY_SECONDS)

print("Video opened successfully")
print(f"Video: {VIDEO_PATH}")
print(f"FPS: {fps:.2f}")
print(f"Total frames: {total_frames}")
print(f"Duration: {duration:.2f} seconds")

all_rgb_pixels = []

frame_index = 0
sampled_frame_count = 0

while True:
    ret, frame_bgr = cap.read()

    if not ret:
        break

    if frame_index % sample_every_n_frames == 0:
        crop_bgr = frame_bgr[y1:y2, x1:x2]

        panel_mask = create_rgb_panel_mask(crop_bgr)

        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)

        panel_pixels = crop_rgb[panel_mask > 0]

        if len(panel_pixels) > 0:
            if len(panel_pixels) > MAX_PIXELS_PER_FRAME:
                selected_indices = np.random.choice(
                    len(panel_pixels),
                    MAX_PIXELS_PER_FRAME,
                    replace=False
                )
                panel_pixels = panel_pixels[selected_indices]

            all_rgb_pixels.append(panel_pixels)

        print(
            f"Sampled frame {sampled_frame_count}: "
            f"time = {frame_index / fps:.1f}s, "
            f"panel pixels = {len(panel_pixels)}"
        )

        sampled_frame_count += 1

    frame_index += 1

cap.release()

if len(all_rgb_pixels) == 0:
    raise RuntimeError("No panel pixels were collected. Check crop and mask settings.")

all_rgb_pixels = np.vstack(all_rgb_pixels)

print(f"Total sampled panel pixels: {len(all_rgb_pixels)}")


# ============================================================
# 6. Convert sampled RGB panel pixels to CIE xy
# ============================================================

xy = srgb_to_xy(all_rgb_pixels)

print(f"Valid CIE xy points: {len(xy)}")


# ============================================================
# 7. Compute representative colour points
# ============================================================

mean_rgb = np.mean(all_rgb_pixels, axis=0)
median_rgb = np.median(all_rgb_pixels, axis=0)
p10_rgb = np.percentile(all_rgb_pixels, 10, axis=0)
p90_rgb = np.percentile(all_rgb_pixels, 90, axis=0)

representative_rgb = np.vstack([
    mean_rgb,
    median_rgb,
    p10_rgb,
    p90_rgb
])

representative_xy = srgb_to_xy(representative_rgb)

mean_xy = representative_xy[0]
median_xy = representative_xy[1]
p10_xy = representative_xy[2]
p90_xy = representative_xy[3]


# ============================================================
# 8. Plot CIE 1931 chromaticity diagram
# ============================================================

fig, ax = plot_chromaticity_diagram_CIE1931(
    show=False
)

# Plot sampled panel colour points
ax.scatter(
    xy[:, 0],
    xy[:, 1],
    s=2,
    alpha=0.08,
    label="Sampled DJI_2 panel pixels"
)

# Plot representative points
ax.scatter(
    mean_xy[0],
    mean_xy[1],
    s=80,
    marker="o",
    label="Mean panel colour"
)

ax.scatter(
    median_xy[0],
    median_xy[1],
    s=80,
    marker="s",
    label="Median panel colour"
)

ax.scatter(
    p10_xy[0],
    p10_xy[1],
    s=80,
    marker="^",
    label="P10 panel colour"
)

ax.scatter(
    p90_xy[0],
    p90_xy[1],
    s=80,
    marker="v",
    label="P90 panel colour"
)

# sRGB primaries
sRGB_primaries = np.array([
    [0.64, 0.33],
    [0.30, 0.60],
    [0.15, 0.06],
    [0.64, 0.33]
])

ax.plot(
    sRGB_primaries[:, 0],
    sRGB_primaries[:, 1],
    label="sRGB gamut"
)

# D65 white point
D65 = np.array([0.3127, 0.3290])

ax.scatter(
    D65[0],
    D65[1],
    s=90,
    marker="x",
    label="D65 white point"
)

ax.set_title("DJI_2 RGB panel colours on CIE 1931 chromaticity diagram")
ax.set_xlabel("CIE x")
ax.set_ylabel("CIE y")
ax.legend(loc="upper right")
ax.grid(True)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=300)
plt.show()

print(f"Saved CIE chromaticity plot to: {OUTPUT_PATH}")