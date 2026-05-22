# 34241-DVT-Project
Code Repository for DVT project 
# Frame Quality and Usability Assessment for Drone Based PV Inspection

This repository contains the project work for the Digital Video Technology course. The project evaluates whether frames extracted from drone videos can be used for photovoltaic module inspection. The focus is not on final defect diagnosis, but on assessing whether inspection-suitable frames can be selected from thermal and RGB drone videos.

## Project Objective

Drone inspections often rely on still images, especially for thermal inspection of PV modules. However, drone videos provide multiple candidate frames from the same inspection scene. This project investigates whether video frames can be filtered and ranked to identify frames suitable for inspection.

The main objective is to evaluate:

1. Frame sharpness and blur
2. Panel coverage
3. Junctions and obstructions
4. Relative thermal intensity behaviour
5. Hotspot-like contrast in thermal video
6. Visual usability in RGB video
7. Usable frame ratio across sampled video frames

## Dataset

Two drone videos are used in this project:

| Video | Modality | Purpose |
|---|---|---|
| DJI_1 | Thermal video | Relative thermographic frame analysis |
| DJI_2 | RGB video | Visual frame usability analysis |

The thermal video is colour mapped, so pixel values are treated as relative thermal intensity values and not calibrated temperatures.

The RGB video is used for visual frame analysis, including panel coverage, sharpness, junction detection, and smooth non-grid object rejection.

## Repository Structure

```text
Project/
│
├── DJI_1.mp4
├── DJI_1.SRT
├── DJI_2.mp4
├── DJI_2.SRT
│
├── analyse_DJI_1_thermal_video_v2.py
├── analyse_DJI_2_rgb_final.py
├── plot_DJI_2_CIE_chromaticity.py
│
├── v2_frames_DJI_1/
├── v2_cropped_frames_DJI_1/
├── v2_panel_masks_DJI_1/
├── v2_hotspot_masks_DJI_1/
├── v2_hotspot_overlays_DJI_1/
├── v2_results_DJI_1/
├── v2_plots_DJI_1/
│
├── frames_DJI_2/
├── cropped_frames_DJI_2/
├── panel_masks_DJI_2/
├── hotspot_masks_DJI_2/
├── hotspot_overlays_DJI_2/
├── results_DJI_2/
├── plots_DJI_2/
│
└── report/
