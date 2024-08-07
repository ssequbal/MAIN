import glob
import os
import cv2
import numpy as np
import argparse

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

     # Create "detected" folder to save images with detected corners
    detected_folder = os.path.join(image_dir, "detected")
    if not os.path.exists(detected_folder):
        os.makedirs(detected_folder)
 
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
            output_filename = os.path.join(detected_folder, f"detected_corners_{found_count}.png")
            cv2.imwrite(output_filename, img)
            print(f"Detected corners in image {found_count}")

    if found_count == 0:
        raise ValueError("No chessboard corners found in any image. Calibration failed.")

    # Calibrate the camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, gray_img.shape[::-1], None, None
    )

    print(f"Performed calibration on {found_count} images")
    print(f"Re-projection error: {ret}")
    print(f"Camera matrix: \n{mtx}")
    print(f"Distortion coefficients: \n{dist}")

    # Extract radial and tangential distortion coefficients
    k1, k2, p1, p2, k3 = dist[0][:5]
    print(f"Radial distortion coefficients: k1={k1}, k2={k2}, k3={k3}")
    print(f"Tangential distortion coefficients: p1={p1}, p2={p2}")

    # Save calibration results
    save_dir = "calibration"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    np.save(os.path.join(save_dir, "cam_matrix.npy"), mtx)
    np.save(os.path.join(save_dir, "distortion_matrix.npy"), dist)
    np.save(os.path.join(save_dir, "translation_vectors.npy"), tvecs)
    np.save(os.path.join(save_dir, "rotation_vectors.npy"), rvecs)

    return mtx, dist, (k1, k2, k3), (p1, p2)

# Example usage
#camera_matrix, distortion_coefficients, radial_distortion, tangential_distortion = calibrate_camera("calibration_images", 8, 6, 0.02)
def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Camera calibration script")

    parser.add_argument("--checkerboard", action="store_true")



    args = parser.parse_args()

    if args.checkerboard:
        calibrate_camera("calibration_images", 8, 6, 0.787)
if __name__ == "__main__":
    main()