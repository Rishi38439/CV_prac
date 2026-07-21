import os
import sys
import cv2
import numpy as np

# 1. Load image relative to script location
path = "/home/itl7/CV_prac/Exp2/3533916.jpg"

image = cv2.imread(path)
if image is None:
    print('Image not found – check the file path!')
    sys.exit()

corner_img = image.copy()
interest_img = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- 1. Harris Corner Detector ---
gray_float = np.float32(gray)
harris = cv2.cornerHarris(gray_float, blockSize=2, ksize=3, k=0.04,thickness=3)
# Increase dilation kernel to make Harris dots larger/bolder
kernel = np.ones((5, 5), np.uint8)
harris = cv2.dilate(harris, kernel)
corner_img[harris > 0.01 * harris.max()] = [0, 0, 255]

# --- 2. Shi-Tomasi Corner Detector ---
corners = cv2.goodFeaturesToTrack(gray, maxCorners=150, qualityLevel=0.01, minDistance=10)
if corners is not None:
    corners = np.int32(corners)
    for c in corners:
        x, y = c.ravel()
        # --- INCREASED DOT SIZE & BORDER ---
        # radius=10 makes the dot bigger; thickness=3 adds a bold border (or use -1 for filled)
        cv2.circle(interest_img, (x, y), radius=10, color=(0, 255, 0), thickness=3)

# --- 3. Blob Detector ---
params = cv2.SimpleBlobDetector_Params()
params.filterByArea = True
params.minArea = 80
params.filterByCircularity = False
params.filterByConvexity = False
params.filterByInertia = False

detector = cv2.SimpleBlobDetector_create(params)
keypoints = detector.detect(gray)

# --- INCREASE BLOB CIRCLE SIZE ---
# Multiply keypoint size to make the drawn blob circles larger on screen
for kp in keypoints:
    kp.size *= 2.5  # Increase size multiplier as needed (e.g., 2.0, 3.0)

# Draw blobs onto image
blob_img = cv2.drawKeypoints(
    image, 
    keypoints, 
    None, 
    (255, 0, 0), 
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)

# --- Save Results ---
cv2.imwrite("Harris_Corners.jpg", corner_img)
cv2.imwrite("Interest_Points.jpg", interest_img)
cv2.imwrite("Blob_Detection.jpg", blob_img)

print("Saved updated images with larger dots and thicker borders successfully!")