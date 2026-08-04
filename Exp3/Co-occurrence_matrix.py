import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops

# 1. Define image file paths (replace with your actual file names)
image_paths = ["images/sat.jpg", "images/agri.jpg", "imaleg/shapes.jpg"]

# 2. Configure GLCM parameters
# Distance between pixel pairs (e.g., 1 pixel apart)
distances = [1] 
# Angles in radians: 0, 45, 90, 135 degrees
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] 

for idx, path in enumerate(image_paths, start=1):
    # Load image in grayscale (GLCM requires single-channel 8-bit images)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not load image from {path}. Check the file path.")
        continue
        
    # 3. Compute GLCM
    # levels=256 specifies standard 8-bit grayscale intensity levels
    glcm = graycomatrix(img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    print(f"\n--- Results for Image {idx}: {path} ---")
    print(f"GLCM shape: {glcm.shape} (levels, levels, distances, angles)")
    
    # 4. Extract Texture Properties from GLCM
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    
    print(f"Contrast:    {contrast:.4f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"Energy:      {energy:.4f}")
    print(f"Homogeneity: {homogeneity:.4f}")

    # Optional: Display the original image
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.title(f"Image {idx}")
    plt.axis('off')
    plt.show()
