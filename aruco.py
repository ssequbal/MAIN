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
                    marker_size=displayMarkerSize
                else:
                    marker_size=offsetMarkerSize
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

    return rvec_marker_4, tvec_marker_5-tvec_marker_4


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


            return rvec, tvec-diff_tvec
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
    green_pixel=np.load("green_dot_center_transforms.npy") 

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
        
        # Negative sign is added in the y direction due to the different coordinate system from the green pixel approach
        diff_tvec=[green_pixel[0],-green_pixel[1],0]
        
        rvec,tvec=detect_single_marker("aruco_image/main.jpg",aruco_dict_types,mtx,dist,diff_rvec,diff_tvec)

        save_dir = "final"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        distortion_file_path = os.path.join(save_dir, "final_values.txt")
        with open(distortion_file_path, "w") as f:
            f.write(f"flip signs for x and y for simulator\n")
            f.write(f"Eye:\n")
            f.write(f"{tvec}")
            f.write(f"\nLookAt:\n")
            f.write(f"{0,0,0}\n")
            f.write(f"Up\n")
            f.write(f"[{0,-1,0}]\n")


if __name__ == "__main__":
    main()
   