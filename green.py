import cv2
import numpy as np
import cv2.aruco as aruco


# Function to detect the green dot, ArUco marker, and calculate the distance in translation matrix
def detect_green_dot_and_aruco(image_path, output_path, camera_matrix, dist_coeffs):
    image = cv2.imread(image_path)

    # Define range for green color in RGB
    lower_green = np.array([0, 100, 0])
    upper_green = np.array([80, 255, 80])

    # Threshold the RGB image to get only green colors
    mask = cv2.inRange(image, lower_green, upper_green)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Draw contours on the original image and get the centroid of the largest contour
    max_contour = max(contours, key=cv2.contourArea)
    M = cv2.moments(max_contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    else:
        cX, cY = 0, 0
    cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 2)
    cv2.circle(image, (cX, cY), 7, (255, 0, 0), -1)

    # Detect ArUco marker
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_100)
    parameters = aruco.DetectorParameters_create()
    corners, ids, _ = aruco.detectMarkers(image, aruco_dict, parameters=parameters)

    if len(corners) > 0:
        # Draw ArUco markers
        aruco.drawDetectedMarkers(image, corners, ids)

        # Define the real world coordinates of the ArUco marker corners (assuming a square marker)
        marker_length = 0.05  # marker side length in meters
        obj_points = np.array(
            [
                [-marker_length / 2, marker_length / 2, 0],
                [marker_length / 2, marker_length / 2, 0],
                [marker_length / 2, -marker_length / 2, 0],
                [-marker_length / 2, -marker_length / 2, 0],
            ]
        )

        # Estimate the pose of the ArUco marker
        ret, rvec, tvec = cv2.solvePnP(
            obj_points, corners[0], camera_matrix, dist_coeffs
        )

        if ret:
            # Project the green dot centroid into the 3D space
            image_points = np.array([[cX, cY]], dtype=np.float32)
            object_points = cv2.undistortPoints(
                image_points, camera_matrix, dist_coeffs
            )
            object_points = np.append(
                object_points, [[1]], axis=1
            )  # Add z=1 for homogenous coordinates

            # Compute the transformation matrix from the rotation and translation vectors
            rotation_matrix, _ = cv2.Rodrigues(rvec)
            transformation_matrix = np.hstack((rotation_matrix, tvec))

            # Transform the green dot centroid into the marker coordinate system
            object_points_transformed = transformation_matrix @ object_points.T
            distance_vector = object_points_transformed[:3] - tvec

            print(
                f"Distance vector from ArUco marker to green dot in meters: {distance_vector.flatten()}"
            )

    else:
        print("No ArUco markers detected")

    # Save the image with contours and ArUco markers
    cv2.imwrite(output_path, image)
    print(f"Image with contours and ArUco markers saved to {output_path}")


# Example usage:
input_image_path = "green_dot_image/green_dot.jpg"
output_image_path = "green_dot_with_contours_and_aruco.jpg"

# Example camera matrix and distortion coefficients (you need to replace these with your own calibration data)
camera_matrix = np.array([[1000, 0, 320], [0, 1000, 240], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion

detect_green_dot_and_aruco(
    input_image_path, output_image_path, camera_matrix, dist_coeffs
)
