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


    # Extract radial and tangential distortion coefficients
    k1, k2, p1, p2, k3 = dist[0][:5]
    # Write the radial and tangential distortion coefficients to a file
    save_dir = "calibration"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    distortion_file_path = os.path.join(save_dir, "distortion_coefficients.txt")
    with open(distortion_file_path, "w") as f:
        f.write(f"Radial distortion coefficients:\n")
        f.write(f"k1={k1}\n")
        f.write(f"k2={k2}\n")
        f.write(f"k3={k3}\n")
        f.write(f"\nTangential distortion coefficients:\n")
        f.write(f"p1={p1}\n")
        f.write(f"p2={p2}\n")


    
    print(f"Radial and tangential distortion coefficients saved to {distortion_file_path}")

    # Save calibration results
    np.save(os.path.join(save_dir, "cam_matrix.npy"), mtx)
    np.save(os.path.join(save_dir, "distortion_matrix.npy"), dist)


    return mtx, dist, (k1, k2, k3), (p1, p2)



def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Calibration script")
    parser.add_argument("--image_path", type=str,default="calibration_images")
    parser.add_argument('--column_count', type=int, default=8)
    parser.add_argument('--row_count', type=int, default=6)
    parser.add_argument('--size', type=int, default=0.02)

    args = parser.parse_args()

    calibrate_camera(args.image_path, args.column_count, args.row_count,args.size)


if __name__ == "__main__":
    main()