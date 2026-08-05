import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.feature import graycomatrix, graycoprops

# 1. Define image file paths (replace with your actual file names)
image_paths = ["images/sat.png", "images/agri.jpg", "images/shapes.png"]

# 2. Configure GLCM parameters
# Distance between pixel pairs (e.g., 1 pixel apart)
distances = [1] 
# Angles in radians: 0, 45, 90, 135 degrees
angles = [0, np.pi/4, np.pi/2, 3*np.pi/4] 

output_dir = "grayscale_images"
os.makedirs(output_dir, exist_ok=True)

results = []

for idx, path in enumerate(image_paths, start=1):
    # Load image in grayscale (GLCM requires single-channel 8-bit images)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Error: Could not load image from {path}. Check the file path.")
        continue

    gray_filename = os.path.splitext(os.path.basename(path))[0] + "_grayscale.png"
    gray_path = os.path.join(output_dir, gray_filename)
    cv2.imwrite(gray_path, img)
        
    # 3. Compute GLCM
    # levels=256 specifies standard 8-bit grayscale intensity levels
    glcm = graycomatrix(img, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    
    # 4. Extract Texture Properties from GLCM
    contrast = graycoprops(glcm, 'contrast')[0]
    correlation = graycoprops(glcm, 'correlation')[0]
    energy = graycoprops(glcm, 'energy')[0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0]

    results.append({
        "Image": f"Image {idx}",
        "Grayscale File": gray_path,
        "Contrast": float(np.mean(contrast)),
        "Correlation": float(np.mean(correlation)),
        "Energy": float(np.mean(energy)),
        "Homogeneity": float(np.mean(homogeneity)),
    })

    # Optional: Display the original image
    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='gray')
    plt.title(f"Image {idx}")
    plt.axis('off')
    plt.show()

print("\nGLCM feature table:")
print(f"{'Image':<10} {'Contrast':>10} {'Correlation':>12} {'Energy':>10} {'Homogeneity':>14}")
for row in results:
    print(
        f"{row['Image']:<10} {row['Contrast']:>10.4f} {row['Correlation']:>12.4f} "
        f"{row['Energy']:>10.4f} {row['Homogeneity']:>14.4f}"
    )

fig, ax = plt.subplots(figsize=(10, 2.5))
ax.axis('off')
table_data = [
    [row['Image'], f"{row['Contrast']:.4f}", f"{row['Correlation']:.4f}", f"{row['Energy']:.4f}", f"{row['Homogeneity']:.4f}"]
    for row in results
]
table = ax.table(
    cellText=table_data,
    colLabels=['Image', 'Contrast', 'Correlation', 'Energy', 'Homogeneity'],
    loc='center',
    cellLoc='center'
)
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 1.4)
plt.tight_layout()
plt.show()
