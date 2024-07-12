import cv2
import pywt
import numpy as np


def w2d(img, mode="haar", level=1):
    image = img
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    modified = np.float32(gray_image)
    modified /= 255
    coef = pywt.wavedec2(modified, mode, level)

    coeffs_H = list(coef)
    coeffs_H[0] *= 0

    image_H = pywt.waverec2(coeffs_H, mode)
    image_H *= 255
    image_H = np.uint8(image_H)

    return image_H
