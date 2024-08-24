import cv2
import numpy as np
from calibration import calibrate_camera
import argparse
import os 
from configparser import ConfigParser
import time

# Read from the configuration file
config = ConfigParser()
config.read("config.txt")

# Read marker size values
displayMarkerSize = config.getfloat("Aruco", "displayMarkerSize")  
offsetMarkerSize = config.getfloat("Aruco", "offsetMarkerSize")  

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
        parameters = cv2.aruco.DetectorParameters()

        corners, ids, rejected_img_points = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if len(corners) > 0:
            for i in range(len(ids)):
                if ids[i] == 4:
                    marker_size=offsetMarkerSize
                else:
                    marker_size=displayMarkerSize
                rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(corners[i],marker_size,matrix_coefficients,distortion_coefficients)

                print(f"Marker ID {ids[i]} - Translation Vector: {tvec}, Rotation Vector: {rvec}")
                cv2.aruco.drawDetectedMarkers(frame, corners)
                translation_vectors.append(tvec[0])  # Extracting the translation vector from the nested structure
                rotation_vectors.append(rvec[0])  # Extracting the rotation vector from the nested structure

                # Check for Marker ID 4 and Marker ID 5
                if ids[i] == 4:
                    tvec_marker_4 = tvec[0][0]
                    rvec_marker_4 = rvec[0][0]
                elif ids[i] == 5:
                    tvec_marker_5 = tvec[0][0]
                    rvec_marker_5 = rvec[0][0]

            cv2.imwrite("output_image_with_markers.jpg", frame)

    return rvec_marker_5-rvec_marker_4, tvec_marker_5-tvec_marker_4


# Function used to calculate the Tvec and Rvec between the single aruco marker and the camera with adjusted values
def detect_single_marker(image_path,aruco_dict_types,matrix_coefficients,distortion_coefficients,diff_rvec,diff_tvec,marker_size=0.025):
    frame = cv2.imread(image_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for aruco_dict_type in aruco_dict_types:

        aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_type)
        parameters = cv2.aruco.DetectorParameters()

        corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

        if len(corners) > 0:
            rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners[0], marker_size, matrix_coefficients, distortion_coefficients)
            print(f"Detected Marker - Translation Vector: {tvec}, Rotation Vector: {rvec}")
            cv2.aruco.drawDetectedMarkers(frame, corners)

            cv2.imwrite("DetectedArUcoMarker.jpg", frame)


            return rvec-diff_rvec , tvec- diff_tvec
        else:
            print("Error: No markers found.")
            return None, None


aruco_dict_types = [cv2.aruco.DICT_6X6_100, cv2.aruco.DICT_5X5_100]


def main():

    parser = argparse.ArgumentParser(description="Camera calibration script")

    parser.add_argument("--aruco_image", action="store_true", help="Find transformation and rotation offset between the two aruco markers")
    parser.add_argument("--main_image", action="store_true", help="Run main calibration")

    args = parser.parse_args()
    mtx = np.load("calibration/cam_matrix.npy")
    dist = np.load("calibration/distortion_matrix.npy")

    # Save calibration results
    save_dir = "difference_mtx"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if args.aruco_image:
        aruco_dict_types = [ cv2.aruco.DICT_6X6_100,cv2.aruco.DICT_5X5_100]
        rvec_marker_difference, tvec_marker_difference = pose_estimation("aruco_image/aruco.jpg", aruco_dict_types, mtx, dist)
        np.save(os.path.join(save_dir, "difference_tvec_matrix.npy"), tvec_marker_difference)
        np.save(os.path.join(save_dir, "difference_rvec_matrix.npy"), rvec_marker_difference)

    if args.main_image:
        aruco_dict_types = [cv2.aruco.DICT_6X6_100]

        diff_rvec = np.load("difference_mtx/difference_rvec_matrix.npy")
        diff_tvec = np.load("difference_mtx/difference_tvec_matrix.npy") 
        
        rvec,tvec=detect_single_marker("aruco_image/main.jpg",aruco_dict_types,mtx,dist,diff_rvec,diff_tvec)

        print("diff",tvec,rvec)

        # # Extract position (translation vector)
        # position = tvec[0][0]


        # # Convert rotation vector to rotation matrix
        # rotation_matrix, _ = cv2.Rodrigues(rvec[0][0])

        # # Convert rotation matrix to Euler angles (X, Y, Z rotation)
        # def rotation_matrix_to_euler_angles(R):
        #     sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
        #     singular = sy < 1e-6

        #     if not singular:
        #         x = np.arctan2(R[2, 1], R[2, 2])
        #         y = np.arctan2(-R[2, 0], sy)
        #         z = np.arctan2(R[1, 0], R[0, 0])
        #     else:
        #         x = np.arctan2(-R[1, 2], R[1, 1])
        #         y = np.arctan2(-R[2, 0], sy)
        #         z = 0

        #     return np.array([x, y, z])

        # orientation = rotation_matrix_to_euler_angles(rotation_matrix)

        # print("Position (X, Y, Z):", position)
        # print("Orientation (rotation around X, Y, Z):", orientation)



if __name__ == "__main__":
    main()
   