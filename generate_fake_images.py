import cv2
import numpy as np
import os

os.makedirs("training/buenas", exist_ok=True)
os.makedirs("training/defectuosas", exist_ok=True)

for i in range(10):
    img_good = np.full((128,128,3), 200, np.uint8)
    img_bad = img_good.copy()
    cv2.circle(img_bad, (64,64), 30, (0,0,0), -1)

    cv2.imwrite(f"training/buenas/good_{i}.jpg", img_good)
    cv2.imwrite(f"training/defectuosas/bad_{i}.jpg", img_bad)

print("Imágenes generadas.")
