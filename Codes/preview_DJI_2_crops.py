import cv2
from pathlib import Path

PROJECT_DIR = Path(
    r"D:\M.Sc. Autonomous Systems - DTU\Spring Semester\34241 Digital video technology\Project"
)

FRAME_PATH = PROJECT_DIR / "frames_DJI_2" / "frame_0005_t_9.8s.jpg"
OUT_DIR = PROJECT_DIR / "crop_preview_DJI_2"
OUT_DIR.mkdir(exist_ok=True)

frame = cv2.imread(str(FRAME_PATH))

if frame is None:
    raise FileNotFoundError(f"Could not read frame: {FRAME_PATH}")

h, w = frame.shape[:2]
print(f"Frame size: width={w}, height={h}")

crop_options = {
    "crop_A": (0, 690, w, 1500),
    "crop_B": (0, 690, w, 1600),
    "crop_C": (0, 690, w, 1700),
    "crop_D": (0, 690, w, 1800),
    "crop_E": (0, 690, w, 1550),
}

for name, (x1, y1, x2, y2) in crop_options.items():
    crop = frame[y1:y2, x1:x2]
    out_path = OUT_DIR / f"{name}.jpg"
    cv2.imwrite(str(out_path), crop)
    print(f"Saved {out_path}")

print("Crop previews created successfully.")