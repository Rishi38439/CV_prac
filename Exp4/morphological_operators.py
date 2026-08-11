import os

import cv2
import numpy as np
import matplotlib


matplotlib.use("Agg")
import matplotlib.pyplot as plt


script_dir = os.path.dirname(os.path.abspath(__file__))
image_names = ["3533916.jpg", "agri.jpeg", "salimg.jpeg"]
kernel = np.ones((5, 5), np.uint8)

fig, axes = plt.subplots(len(image_names), 5, figsize=(18, 10))
if len(image_names) == 1:
    axes = np.array([axes])

for row, image_name in enumerate(image_names):
    image_path = os.path.join(script_dir, image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise FileNotFoundError(f"Could not load image: {image_path}")

    eroded = cv2.erode(image, kernel, iterations=1)
    dilated = cv2.dilate(image, kernel, iterations=1)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    panels = [
        (image, "Original"),
        (eroded, "Erosion"),
        (dilated, "Dilation"),
        (opened, "Opening"),
        (closed, "Closing"),
    ]

    for col, (panel, title) in enumerate(panels):
        axis = axes[row, col]
        axis.imshow(panel, cmap="gray")
        axis.set_title(f"{image_name}\n{title}")
        axis.axis("off")


plt.tight_layout()
output_path = os.path.join(script_dir, "morphological_operators_result.png")
plt.savefig(output_path, dpi=200, bbox_inches="tight")
print(f"Saved morphology results to: {output_path}")