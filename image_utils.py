from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    # טעינת התמונה והפיכתה למערך נומפאי
    img = Image.open(path)
    return np.array(img)

def edge_detection(image):
    # המרה לגרייסקייל אם זה עדיין צבעוני
    if image.ndim == 3:
        gray_image = np.mean(image, axis=2)
    else:
        gray_image = image

    # פילטרים (Kernels) לזיהוי קצוות
    kernelY = np.array([[ 1,  2,  1], [ 0,  0,  0], [-1, -2, -1]])
    kernelX = np.array([[-1,  0,  1], [-2,  0,  2], [-1,  0,  1]])

    # ביצוע הקונבולוציה
    edgeY = convolve2d(gray_image, kernelY, mode='same', boundary='fill', fillvalue=0)
    edgeX = convolve2d(gray_image, kernelX, mode='same', boundary='fill', fillvalue=0)

    # חישוב עוצמת הקצוות
    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
