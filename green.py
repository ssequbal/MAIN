import cv2
import numpy as np

# Load the image
image = cv2.imread('green.jpg')

# Convert the image to grayscale for Aruco detection
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the Aruco dictionary
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_100)
parameters = cv2.aruco.DetectorParameters()

# Detect the Aruco markers in the image
corners, ids, rejected = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

# Assuming there is only one Aruco marker and taking the first one detected
if ids is not None and len(corners) > 0:
    # Get the center of the Aruco marker
    aruco_center = np.mean(corners[0][0], axis=0)
    aruco_center = tuple(aruco_center.astype(int))
    cv2.circle(image, aruco_center, 5, (0, 0, 255), -1)  # Draw the center for visualization

    # Calculate the real-world distance per pixel using the known Aruco marker size
    marker_size = 0.025  
    pixel_per_meter = cv2.norm(corners[0][0][0], corners[0][0][1]) / marker_size

    # Detect the green dot
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
   
    # Find contours of the green dot
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
   
    if contours:
        # Assume the largest contour is the green dot
        green_dot_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(green_dot_contour)
        if M['m00'] != 0:
            green_dot_center = (int(M['m10'] / M['m00']), int(M['m01'] / M['m00']))
            cv2.circle(image, green_dot_center, 5, (255, 0, 0), -1)  # Draw the center for visualization
           
            # Calculate distances in x and y coordinates
            x_distance = abs(aruco_center[0] - green_dot_center[0])
            y_distance = abs(aruco_center[1] - green_dot_center[1])
           
            # Convert pixel distances to real-world distances
            real_x_distance = x_distance / pixel_per_meter
            real_y_distance = y_distance / pixel_per_meter
           
            print(f"Pixel Distance in X: {x_distance:.2f} pixels")
            print(f"Pixel Distance in Y: {y_distance:.2f} pixels")
            print(f"Real World Distance in X: {real_x_distance:.2f} meters")
            print(f"Real World Distance in Y: {real_y_distance:.2f} meters")
        else:
            print("Could not find the center of the green dot.")
    else:
        print("Green dot not found.")
else:
    print("Aruco marker not found.")

# Display the image with marked centers
cv2.imshow('Image with Aruco and Green Dot', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
