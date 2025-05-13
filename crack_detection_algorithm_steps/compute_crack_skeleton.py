import numpy as np
import cv2
from skimage.morphology import skeletonize

def compute_skeleton(contours, shape):
    binary = np.zeros(shape, dtype=np.uint8)
    cv2.drawContours(binary, [cnt for cnt, _ in contours], -1, 255, thickness=cv2.FILLED)
    skeleton = skeletonize(binary > 0)
    return skeleton