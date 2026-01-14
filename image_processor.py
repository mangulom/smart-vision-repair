import cv2
import numpy as np

def auto_enhance(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # CLAHE → mejora contraste adaptativo
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)

    # Reducción de ruido
    denoise = cv2.fastNlMeansDenoising(enhanced, h=10)

    return cv2.cvtColor(denoise, cv2.COLOR_GRAY2BGR)
