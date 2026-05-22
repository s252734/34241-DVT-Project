import pandas as pd
import shutil
from pathlib import Path

PROJECT_DIR = Path(r"D:\M.Sc. Autonomous Systems - DTU\Spring Semester\34241 Digital video technology\Project")

FRAME_DIR = PROJECT_DIR / "frames_DJI_1"
RESULT_DIR = PROJECT_DIR / "results_DJI_1"
REVIEW_DIR = PROJECT_DIR / "review_frames_DJI_1"
REVIEW_DIR.mkdir(exist_ok=True)

df = pd.read_csv(RESULT_DIR / "DJI_1_frame_analysis.csv")

candidates = []

candidates.append(("best_sharpness", df.loc[df["laplacian_sharpness"].idxmax()]))
candidates.append(("worst_sharpness", df.loc[df["laplacian_sharpness"].idxmin()]))

median_value = df["laplacian_sharpness"].median()
median_row = df.iloc[(df["laplacian_sharpness"] - median_value).abs().argsort()[:1]].iloc[0]
candidates.append(("median_sharpness", median_row))

candidates.append(("highest_hotspot_contrast", df.loc[df["hotspot_background_contrast"].idxmax()]))
candidates.append(("lowest_hotspot_contrast", df.loc[df["hotspot_background_contrast"].idxmin()]))

for label, row in candidates:
    src = FRAME_DIR / row["frame_file"]
    dst = REVIEW_DIR / f"{label}_{row['frame_file']}"
    shutil.copy(src, dst)

print("Review frames copied to:", REVIEW_DIR)

for label, row in candidates:
    print(label, row["frame_file"], "time:", row["time_sec"], "sharpness:", row["laplacian_sharpness"], "contrast:", row["hotspot_background_contrast"])