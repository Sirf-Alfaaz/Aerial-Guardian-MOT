# Aerial Guardian – Drone Based Multi Object Tracking

A lightweight aerial surveillance pipeline for real-time person detection and multi-object tracking in drone footage using YOLOv8 and ByteTrack.

This project was developed for aerial monitoring scenarios where pedestrians appear extremely small, camera motion is dynamic, and maintaining stable identities across frames is challenging. The pipeline processes image-sequence datasets such as VisDrone MOT sequences and generates tracked output videos with trajectory visualization, FPS monitoring, and stable multi-object tracking.

---

# Problem Statement

Drone-based surveillance introduces several challenges compared to conventional ground-camera tracking systems:

* Extremely small pedestrian targets
* Camera ego-motion from UAV movement
* Frequent occlusions and scale variations
* Motion blur and low-resolution targets
* Real-time deployment constraints

The objective of this project is to engineer a lightweight and practical aerial MOT pipeline capable of balancing:

* detection accuracy,
* tracking stability,
* computational efficiency,
* and edge deployment feasibility.

---

# Pipeline Overview

```text
Input Image Sequence
        ↓
YOLOv8 Person Detection
        ↓
ByteTrack Multi Object Tracking
        ↓
Track Confirmation & Filtering
        ↓
Trajectory Smoothing
        ↓
HUD + FPS Visualization
        ↓
Tracked Output Video
```

---

# Core Features

* YOLOv8 aerial person detection
* ByteTrack-based multi-object tracking
* Stable ID assignment
* Dynamic confidence thresholding
* Trajectory smoothing and motion tails
* FPS benchmarking HUD
* Lightweight real-time architecture
* Image-sequence based inference
* Drone-oriented visualization system

---

# Engineering Optimizations

## 1. Dynamic Confidence Thresholding

Different confidence thresholds are applied based on object position in the frame.

Far-field aerial targets are generally smaller and harder to detect, therefore lower confidence thresholds are used for upper-frame detections while stricter thresholds are applied to near-field detections.

This improves small-object recall while reducing false positives.

---

## 2. Detection Scheduling

To improve runtime performance, detection is not necessarily executed on every frame. Frame scheduling strategies were tested to balance:

* detection quality,
* tracking stability,
* and FPS performance.

This significantly improved runtime efficiency while maintaining acceptable tracking continuity.

---

## 3. Trajectory Smoothing

Detected object centers are temporally smoothed using weighted averaging to reduce jitter caused by:

* noisy detections,
* UAV movement,
* and rapid frame-to-frame variations.

This improves visual consistency and tracking quality.

---

## 4. Lightweight MOT Design

The system intentionally uses:

* YOLOv8s
* ByteTrack

instead of heavier transformer-based or appearance-heavy tracking systems in order to remain lightweight and edge-deployment friendly.

---

# Small Object Detection Strategy

Aerial surveillance datasets contain extremely small pedestrian instances. To improve detection quality:

* Higher inference resolution (`imgsz=1024`) was used
* Dynamic thresholds were applied
* Small false detections were filtered
* Temporal smoothing improved consistency

These optimizations significantly improved aerial tracking quality.

---

# Experimental Approaches Explored

Several advanced approaches were explored during development and benchmarking.

## SAHI Sliced Inference

SAHI (Sliced Aided Hyper Inference) was tested to improve tiny-object detection performance in aerial imagery.

The idea was to:

* slice the frame into smaller regions,
* run high-resolution detection on each slice,
* and improve far-field pedestrian recall.

While SAHI improved tiny-object recall in some scenarios, it introduced several practical challenges:

* significant FPS reduction,
* unstable tracking during rapid drone movement,
* delayed tracker updates,
* and increased computational overhead.

After experimentation, the final submission prioritized a more stable and lightweight real-time pipeline over aggressive sliced inference.

---

## Dynamic Region of Interest (ROI)

Dynamic ROI-based processing was also explored to allocate more computational resources to upper aerial regions where pedestrians appear smaller.

The strategy aimed to:

* apply heavier detection only in far-field regions,
* reduce unnecessary computation in near-field areas,
* and improve small-object detection efficiency.

However, due to rapid drone motion and changing scene dynamics, ROI switching occasionally introduced instability and inconsistent detections.

The final implementation therefore focused on maintaining stable and reliable tracking performance rather than aggressive region-adaptive processing.

---

# Dataset

The project was tested on aerial MOT image sequences from the VisDrone benchmark dataset.

Example sequence structure:

```text
sequences/
    uav0000074_11856_v/
        000001.jpg
        000002.jpg
```

---

# Performance

Example Hardware:

```text
GPU : RTX 4050 Laptop GPU
CPU : Intel i7 13th Gen
RAM : 16GB
```

The final pipeline achieves:

* stable multi-object tracking,
* lightweight runtime performance,
* real-time visualization,
* and consistent tracking IDs.

---

# Output Features

The generated output video includes:

* Bounding boxes
* Persistent track IDs
* Motion trajectory tails
* FPS metrics
* Active object count
* Real-time HUD overlay

---

# Future Improvements

Potential future enhancements include:

* TensorRT acceleration
* FP16 inference optimization
* Jetson deployment
* Camera motion compensation
* Multi-class aerial tracking
* Re-identification based tracking
* Adaptive frame scheduling

---

# Installation

```bash
pip install -r requirements.txt
```

---

# Run

```bash
python scripts/final_pipeline.py
```

---

# Tech Stack

* Python
* OpenCV
* YOLOv8
* ByteTrack
* PyTorch
* NumPy

---

# Conclusion

This project focuses on engineering a practical aerial surveillance pipeline capable of balancing:

* small-object detection,
* multi-object tracking,
* runtime efficiency,
* and deployment feasibility

for real-world UAV surveillance environments.
