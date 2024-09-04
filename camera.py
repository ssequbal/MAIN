import os
import subprocess
import argparse
from configparser import ConfigParser
import time

# Read from the configuration file
config = ConfigParser()
config.read("config.txt")

# Read camera Values
iso = config.getint("Camera", "iso_value")  # Read ISO value from the config file
shutter_speed = config.get("Camera", "shutter_speed")  # Read shutter speed from the config file
aperture_value = config.getfloat("Camera", "aperture_value")  # Read aperture value from the config file


def take_picture(filename, destination, iso, shutter, aperture):
    if not os.path.exists(destination):
        os.mkdir(destination)


    # Setting up the gphoto2 commands
    set_iso_cmd = f"gphoto2 --set-config iso={iso}"
    set_shutter_cmd = f"gphoto2 --set-config shutterspeed={shutter}"
    set_aperture_cmd = f"gphoto2 --set-config aperture={aperture}"
    capture_image_cmd = f"gphoto2 --capture-image-and-download --filename {os.path.join(destination, filename)}"

    # Execute the commands
    try:
        print("Setting ISO:", set_iso_cmd)
        subprocess.run(set_iso_cmd, shell=True, check=True)

        print("Setting Shutter Speed:", set_shutter_cmd)
        subprocess.run(set_shutter_cmd, shell=True, check=True)

        # print("Setting Aperture:", set_aperture_cmd)
        # subprocess.run(set_aperture_cmd, shell=True, check=True)

        print("Capturing Image:", capture_image_cmd)
        result = subprocess.run(capture_image_cmd, shell=True, check=True, capture_output=True, text=True)

        print("Command output:", result.stdout)
        print("Command error:", result.stderr)

    except subprocess.CalledProcessError as e:
        print(f"An error occurred: {e}")
        print("Output:", e.output)
        print("Error:", e.stderr)


def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Camera calibration script")

    parser.add_argument("--calibrate", type=int, help="Number of times to run camera calibration")
    parser.add_argument("--aruco_image", action="store_true", help="Run aruco calibration")
    parser.add_argument("--green", action="store_true", help="Run aruco calibration")
    parser.add_argument("--main", action="store_true", help="Run aruco calibration")
    parser.add_argument("--dark", action="store_true", help="Final Crop")


    args = parser.parse_args()

    if args.calibrate:
        for i in range(args.calibrate):
            raw_image_name = f"{i}.jpg"
            take_picture(raw_image_name, "calibration_images", iso, shutter_speed, aperture_value)

    if args.aruco_image:
        take_picture("aruco.jpg", "aruco_image", iso, shutter_speed, aperture_value)

    if args.green:
       
        take_picture("green.jpg", "green_dot_image", iso, shutter_speed, aperture_value)

    if args.main:
        take_picture("main.jpg", "aruco_image", iso, shutter_speed, aperture_value)

    if args.dark:
        take_picture("dark.jpg", "aruco_image", iso, shutter_speed, aperture_value)

if __name__ == "__main__":
    main()
