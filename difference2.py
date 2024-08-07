import cv2
import numpy as np

def overlay_images(img1, img2, alpha=0.5):
  """Overlays two images with an optional alpha blending factor.

  Args:
    img1: The first image.
    img2: The second image.
    alpha: The blending factor, between 0 and 1.

  Returns:
    The overlaid image.
  """

  # Ensure images have the same shape
  img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))

  # Create an overlayed image
  overlay = cv2.addWeighted(img1, alpha, img2, 1 - alpha, 0)

  return overlay

def show_difference(img1, img2):
  """Calculates and displays the absolute difference between two images.

  Args:
    img1: The first image.
    img2: The second image.
  """
    # Ensure images have the same shape
  img1 = cv2.resize(img1, (img2.shape[1], img2.shape[0]))
  # Convert to grayscale for simpler comparison
  gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
  gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

  # Calculate absolute difference
  diff = cv2.absdiff(gray1, gray2)

  # Enhance contrast for better visibility (optional)
  diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

  cv2.imwrite("Difference.png", diff)
  cv2.waitKey(0)

# Load the two images
img1 = cv2.imread('difference_mat/diff_a.jpg')
img2 = cv2.imread("difference_mat/diff_b.png")

# Overlay images
overlay = overlay_images(img1, img2)
cv2.imwrite("Overlay.png", overlay)

# Show differences
show_difference(img1, img2)

cv2.destroyAllWindows()
