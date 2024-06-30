import glob
import os
import cv2
import numpy as np


def calibrate_camera(image_dir, x_dim, y_dim, square_size):
    # Termination criteria for corner refinement
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)

    # Prepare object points based on the real world coordinates of the checkerboard
    objp = np.zeros((x_dim * y_dim, 3), np.float32)
    objp[:, :2] = np.mgrid[0:x_dim, 0:y_dim].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane

    images = glob.glob(os.path.join(image_dir, "*.jpg"))
    if not images:
        raise FileNotFoundError("No images found in the specified directory")

    found_count = 0
    for filename in images:
        img = cv2.imread(filename)
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray_img, (x_dim, y_dim), None)

        if ret:
            found_count += 1

            # Refine the corner locations
            corners2 = cv2.cornerSubPix(gray_img, corners, (11, 11), (-1, -1), criteria)

            # Append object points and image points
            objpoints.append(objp)
            imgpoints.append(corners2)

            # Draw and display the corners
            cv2.drawChessboardCorners(img, (x_dim, y_dim), corners2, ret)
            cv2.imwrite("detected_corners_" + str(found_count) + ".png", img)
            print(f"Detected corners in image {found_count}")

    if found_count == 0:
        raise ValueError(
            "No chessboard corners found in any image. Calibration failed."
        )

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_img.shape[::-1], None, None
    )

    print(f"Performed calibration on {found_count} images")

    # Save calibration results
    save_dir = "calibration"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, "cam_matrix.npy"), mtx)
    np.save(os.path.join(save_dir, "distortion_matrix.npy"), dist)
    np.save(os.path.join(save_dir, "translation_vectors.npy"), tvecs)
    np.save(os.path.join(save_dir, "rotation_vectors.npy"), rvecs)

    return mtx, dist
