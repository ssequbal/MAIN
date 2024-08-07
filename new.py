import cv2
import numpy as np

# Read input image
img = cv2.imread('final_image.png')

# Mirror in x direction (flip horizontally)
imgX = np.flip(img, axis=1)
# imgX = imgX = img[:, ::-1, :]

# Mirror in y direction (flip vertically)
imgY = np.flip(img, axis=0)
# imgY = img[::-1, :, :]

# Mirror in both directions (flip horizontally and vertically)
imgXY = np.flip(img, axis=(0, 1))
# imgXY = img[::-1, ::-1, :]

# Outputs
cv2.imshow('img', img)
cv2.imshow('imgX', imgX)
cv2.imshow('imgY', imgY)
cv2.imshow('imgXY', imgXY)
cv2.waitKey(0)