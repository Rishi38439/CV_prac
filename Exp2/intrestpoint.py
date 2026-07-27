import os
import sys

import cv2
import numpy as np
import tkinter as tk

from PIL import Image, ImageTk


def resize_to_height(image, target_height):
    height, width = image.shape[:2]
    if height == target_height:
        return image

    scale = target_height / height
    target_width = int(width * scale)
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def label_panel(image, label):
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 42), (0, 0, 0), thickness=-1)
    cv2.putText(
        panel,
        label,
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return panel


# Load image relative to the script location.
script_dir = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(script_dir, "3533916.jpg")

image = cv2.imread(path)
if image is None:
    print("Image not found - check the file path!")
    sys.exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

# CLAHE improves local contrast, which usually makes corner and blob detection cleaner.
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray_enhanced = clahe.apply(gray)

corner_img = image.copy()
interest_img = image.copy()

# --- 1. Harris Corner Detector ---
gray_float = np.float32(gray_enhanced)
harris = cv2.cornerHarris(gray_float, blockSize=3, ksize=5, k=0.04)
harris = cv2.dilate(harris, np.ones((3, 3), np.uint8))
corner_img[harris > 0.02 * harris.max()] = [0, 0, 255]

# --- 2. Shi-Tomasi Corner Detector ---
corners = cv2.goodFeaturesToTrack(
    gray_enhanced,
    maxCorners=150,
    qualityLevel=0.03,
    minDistance=12,
    blockSize=7,
    useHarrisDetector=False,
    k=0.04,
)
if corners is not None:
    corners = np.int32(corners)
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(interest_img, (x, y), radius=4, color=(0, 255, 0), thickness=2)

# --- 3. Blob Detector ---
blob_input = cv2.medianBlur(gray_enhanced, 5)

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 10
params.maxThreshold = 220
params.thresholdStep = 10

params.filterByArea = True
params.minArea = 60
params.maxArea = 5000

params.filterByCircularity = True
params.minCircularity = 0.55

params.filterByConvexity = True
params.minConvexity = 0.75

params.filterByInertia = True
params.minInertiaRatio = 0.25

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(blob_input)

blob_img = cv2.drawKeypoints(
    image,
    keypoints,
    None,
    (255, 0, 0),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
)

# --- Show All Results In One Frame ---
target_height = 520
panels = [
    label_panel(resize_to_height(corner_img, target_height), "Harris Corner Detector"),
    label_panel(resize_to_height(interest_img, target_height), "Shi-Tomasi Corner Detector"),
    label_panel(resize_to_height(blob_img, target_height), "Blob Detector"),
]

combined = cv2.hconcat(panels)
combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

try:
    root = tk.Tk()
    root.title("Interest Point Detectors")

    display_image = Image.fromarray(combined_rgb)
    photo = ImageTk.PhotoImage(display_image)

    label = tk.Label(root, image=photo)
    label.image = photo
    label.pack()

    root.mainloop()
except tk.TclError:
    output_path = os.path.join(script_dir, "interest_points_preview.jpg")
    cv2.imwrite(output_path, combined)
    print(f"No display found. Saved the combined preview to: {output_path}")