import cv2
import numpy as np

def repair_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detectar zonas dañadas automáticamente (threshold)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    # Inpainting clásico
    repaired = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return repaired
