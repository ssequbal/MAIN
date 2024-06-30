import cv2
import numpy as np
import argparse
import os 

# Function used to calculate the Difference in Tvec and Rvec between the two aruco markers
def pose_estimation(image_path,aruco_dict_types,matrix_coefficients,distortion_coefficients,marker_size=0.02):

    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    distance_matrix = None
    translation_vectors = []
    rotation_vectors = []
    tvec_marker_4 = None
    rvec_marker_4 = None
    tvec_marker_5 = None
    rvec_marker_5 = None

    for aruco_dict_type in aruco_dict_types:
        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters_create()

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if len(corners) > 0:
            for i in range(len(ids)):
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i],marker_size,matrix_coefficients,distortion_coefficients)

                print(f"Marker ID {ids[i]} - Translation Vector: {tvec}, Rotation Vector: {rvec}")
                cv2.aruco.drawDetectedMarkers(frame, corners)
                translation_vectors.append(tvec[0])  # Extracting the translation vector from the nested structure
                rotation_vectors.append(rvec[0])  # Extracting the rotation vector from the nested structure

                # Check for Marker ID 4 and Marker ID 5
                if ids[i] == 4:
                    tvec_marker_4 = tvec
                    rvec_marker_4 = rvec
                elif ids[i] == 5:
                    tvec_marker_5 = tvec
                    rvec_marker_5 = rvec

            cv2.imwrite("output_image_with_markers.jpg", frame)

    return rvec_marker_5 - rvec_marker_4, tvec_marker_5 - tvec_marker_4


# Function used to calculate the Tvec and Rvec between the single aruco marker and the camera with adjusted values
def detect_single_marker(image_path,aruco_dict_types,matrix_coefficients,distortion_coefficients,difference_tvec,difference_rvec,marker_size=0.02):
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for aruco_dict_type in aruco_dict_types:

        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters_create()

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if len(corners) > 0:
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], marker_size, matrix_coefficients, distortion_coefficients)
            print(f"Detected Marker - Translation Vector: {tvec}, Rotation Vector: {rvec}")
            cv2.aruco.drawDetectedMarkers(frame, corners)

            cv2.imwrite("DetectedArUcoMarker.jpg", frame)


            return rvec + difference_rvec, tvec + difference_tvec
        else:
            print("Error: No markers found.")
            return None, None


aruco_dict_types = [cv2.aruco.DICT_6X6_100, cv2.aruco.DICT_5X5_100]


def main():

    parser = argparse.ArgumentParser(description="Camera calibration script")

    parser.add_argument("--aruco_image", action="store_true", help="Run aruco calibration")
    parser.add_argument("--main_image", action="store_true", help="Run main calibration")

    args = parser.parse_args()
    mtx = np.load("calibration/cam_matrix.npy")
    dist = np.load("calibration/distortion_matrix.npy")

    # Save calibration results
    save_dir = "difference_mtx"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.aruco_image:
        aruco_dict_types = [cv2.aruco.DICT_6X6_100, cv2.aruco.DICT_5X5_100]
        tvec_marker_difference, rvec_marker_difference = pose_estimation( "aruco_image/aruco.jpg", aruco_dict_types, mtx, dist)
        np.save(os.path.join(save_dir, "difference_tvec_matrix.npy"), tvec_marker_difference)
        np.save(os.path.join(save_dir, "difference_rvec_matrix.npy"), rvec_marker_difference)

        print(tvec_marker_difference, rvec_marker_difference)

    if args.main_image:
        aruco_dict_types = [cv2.aruco.DICT_6X6_100, cv2.aruco.DICT_5X5_100]
        tvec_marker_difference = np.load("difference_mtx/difference_tvec_matrix.npy")
        rvec_marker_difference = np.load("difference_mtx/difference_rvec_matrix.npy")

        detect_single_marker("aruco_image/main.jpg",aruco_dict_types,mtx,dist,tvec_marker_difference,rvec_marker_difference,marker_size=0.02)


if __name__ == "__main__":
    main()
