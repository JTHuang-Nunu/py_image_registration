## Overview

This project implements multi-spectral image alignment and plane coordinate transformation. The primary goal is to align different spectral images to a unified coordinate system using feature point detection, matching, and geometric transformations. This results in accurate overlay and error analysis of multi-spectral images.

<br>

## Features

1. **Core Functionality**:

   - Automatic feature point detection and description (`ORB`, `SURF`).
   - Feature matching using `BF Matcher` and `Flann-based Matcher`.
   - Geometric transformation and image alignment using `Homography`.
   - Error filtering with `RANSAC`.

2. **Applications**:

   - Ideal for remote sensing, multi-spectral image analysis, and any use case requiring image alignment and geometric correction.

3. **Key Techniques**:
   - Extracting significant feature points for accurate image alignment.
   - Applying multiple **OPENCV** matching methods and error filtering techniques to enhance alignment precision.

<br>

## Program Structure

```plaintext
root/
│
├── main.py                          # Main script controlling the overall workflow
├── utils.py                         # Utility functions for image processing and alignment
├── gif_maker.py                     # Module for generating GIFs
├── gif                              # Folder for storing generated GIFs
└── <Image>...                       # Folder for image files
     ├── aligned                     # Aligned spectral images
     ├── report                      # Error and data reports
     ├── aligned_rgb.png             # Stacked RGB images for visualization
     └── ...
```

<br>

## How to Run

Ensure that your Python environment has all required dependencies installed. Run the following command to start the program:

```bash
python main.py
```
