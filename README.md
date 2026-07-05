# Crack Detection in Drainage and City Water Pipelines using Raspberry Pi and Computer Vision

> **Real-Time Edge AI Pipeline Inspection System using Raspberry Pi**

![Python](https://img.shields.io/badge/Python-3.10-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![RaspberryPi](https://img.shields.io/badge/Raspberry%20Pi-4-red)
![YOLO](https://img.shields.io/badge/Detection-Computer%20Vision-orange)
![Status](https://img.shields.io/badge/Project-Completed-success)

---

# Project Overview

Urban drainage and municipal water pipelines are vulnerable to cracks caused by aging infrastructure, excessive pressure, corrosion, and environmental conditions. Manual inspection of underground pipelines is expensive, time-consuming, and often dangerous.

This project presents a **real-time crack detection system** developed using **Raspberry Pi 4** and **Computer Vision**. The system continuously processes images captured from a camera mounted on an inspection robot or mobile platform, detects structural cracks, and highlights them with bounding boxes for immediate identification.

The solution enables low-cost, edge-based infrastructure monitoring without requiring cloud processing, making it suitable for smart city applications.

---

# Problem Statement

Current inspection methods involve

- Manual inspection
- CCTV footage analysis
- Human observation
- High operational costs
- Delayed maintenance

These methods often fail to detect early-stage cracks, leading to

- Water leakage
- Structural failure
- Increased repair costs
- Environmental damage
- Water loss

---

# Proposed Solution

Develop an embedded vision system capable of

- Capturing pipeline images
- Processing images locally on Raspberry Pi
- Detecting visible surface cracks
- Highlighting damaged regions
- Displaying real-time inspection results

The entire processing pipeline runs directly on the Raspberry Pi without requiring external servers.

---

# Objectives

- Detect cracks in drainage pipelines
- Detect cracks in city water pipelines
- Perform real-time image processing
- Reduce inspection time
- Minimize maintenance costs
- Enable preventive maintenance
- Improve smart city infrastructure monitoring

---

# System Architecture

```
Pipeline
     │
     ▼
Camera Module
     │
     ▼
Raspberry Pi 4
     │
     ▼
Image Acquisition
     │
     ▼
Image Preprocessing
     │
     ▼
Edge Detection
     │
     ▼
Contour Extraction
     │
     ▼
Crack Detection Algorithm
     │
     ▼
Bounding Box Generation
     │
     ▼
Display Detection Result
```

---

# Hardware Requirements

| Component | Description |
|------------|------------|
| Raspberry Pi 4 | Edge Computing Platform |
| Pi Camera Module / USB Camera | Image Acquisition |
| MicroSD Card | Operating System |
| Power Supply | Raspberry Pi Power |
| Robot Chassis (Optional) | Pipeline Inspection |
| LED Light (Optional) | Low Light Illumination |

---

# Software Requirements

- Raspberry Pi OS
- Python 3
- OpenCV
- NumPy
- imutils
- VS Code
- Git

---

# Technologies Used

- Raspberry Pi 4
- Python
- OpenCV
- Image Processing
- Edge Computing
- Computer Vision

---

# Image Processing Pipeline

## 1. Image Capture

Frames are captured continuously using the Pi Camera.

↓

## 2. Grayscale Conversion

Converts RGB image into grayscale.

↓

## 3. Gaussian Blur

Removes image noise before edge extraction.

↓

## 4. Edge Detection

Canny Edge Detection identifies crack boundaries.

↓

## 5. Morphological Operations

Removes unwanted noise and enhances crack regions.

↓

## 6. Contour Detection

Contours corresponding to possible cracks are extracted.

↓

## 7. Bounding Box Generation

Detected cracks are highlighted using rectangles.

↓

## 8. Display Result

The processed frame is displayed in real time.

---

# Detection Workflow

```
Input Image

↓

Resize

↓

Grayscale

↓

Gaussian Blur

↓

Canny Edge Detection

↓

Morphological Processing

↓

Find Contours

↓

Filter Noise

↓

Draw Bounding Boxes

↓

Output Detection
```

---

# Project Features

- Real-time crack detection
- Raspberry Pi deployment
- Lightweight processing
- Edge computing
- Automatic defect localization
- Bounding box visualization
- Low-cost implementation
- Smart city compatible
- Portable system
- Easy deployment

---

# Development Stages

## Phase 1

Project Planning

- Literature Survey
- Problem Identification
- Hardware Selection

---

## Phase 2

Hardware Setup

- Raspberry Pi Installation
- Camera Configuration
- Operating System Setup

---

## Phase 3

Software Development

- Python Environment
- OpenCV Installation
- Camera Testing

---

## Phase 4

Algorithm Development

- Image Preprocessing
- Edge Detection
- Contour Analysis
- Crack Identification

---

## Phase 5

Testing

- Indoor Testing
- Outdoor Testing
- Multiple Crack Samples
- Performance Evaluation

---

## Phase 6

Optimization

- Faster Processing
- Noise Reduction
- Better Detection Accuracy

---

# Advantages

- Low-cost inspection
- Portable solution
- Fast processing
- Real-time monitoring
- Easy maintenance
- Preventive fault detection
- Reduced manpower
- Improved infrastructure safety

---

# Applications

- Municipal Water Pipelines
- Drainage Inspection
- Underground Sewer Systems
- Smart Cities
- Water Supply Departments
- Civil Infrastructure
- Industrial Pipelines

---

# Limitations

- Surface cracks only
- Camera lighting dependency
- Performance affected by muddy water
- Requires visible crack exposure

---

# Future Improvements

- YOLO-based crack detection
- Deep Learning segmentation
- Thermal camera integration
- Automatic GPS localization
- Cloud dashboard
- IoT monitoring
- Drone-assisted inspection
- Robotic autonomous navigation
- AI severity classification
- Crack size estimation

---

# Performance

| Parameter | Value |
|------------|---------|
| Platform | Raspberry Pi 4 |
| Programming Language | Python |
| Processing | Edge Computing |
| Detection | Real-Time |
| Visualization | Bounding Boxes |
| Deployment | Embedded |

---

# Learning Outcomes

Through this project, the following skills were developed:

- Embedded Systems
- Raspberry Pi Development
- Computer Vision
- Image Processing
- OpenCV
- Python Programming
- Real-Time Processing
- Edge AI Concepts
- Camera Integration
- Smart Infrastructure Monitoring

---

# Repository Structure

```
Crack-Detection/
│
├── dataset/
├── images/
├── outputs/
├── models/
├── src/
│   ├── detect.py
│   ├── preprocessing.py
│   ├── contour.py
│   ├── utils.py
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

# Installation

```bash
git clone https://github.com/yourusername/Crack-Detection.git

cd Crack-Detection

pip install -r requirements.txt
```

---

# Run

```bash
python detect.py
```

---

# Results

The system successfully detects visible cracks in drainage and municipal water pipelines by processing camera frames in real time and highlighting damaged regions using bounding boxes. The implementation demonstrates the feasibility of deploying lightweight computer vision algorithms on Raspberry Pi for practical infrastructure inspection.

---

# Skills Demonstrated

- Raspberry Pi Development
- Python Programming
- Computer Vision
- OpenCV
- Embedded Systems
- Image Processing
- Edge Computing
- Real-Time Systems
- Camera Integration

---

# Author

**Kanishkar Sengottaiyan**

Electronics and Communication Engineering

Rathinam Technical Campus, Coimbatore

---

# License

This project is developed for academic and research purposes.

