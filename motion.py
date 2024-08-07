
import time
from configparser import ConfigParser
import argparse
import subprocess
from serial import Serial

# Read from the configuration file
config = ConfigParser()
config.read("config.txt")

# Read serial settings
serial_port = config.get("Serial", "serial_port")
baud_rate = config.getint("Serial", "baud_rate")

# Read motion settings
num_iterations = config.getint("Motion", "num_iterations")
direction = config.get("Motion", "direction")
displacement_value = config.getint("Motion", "displacement_value")
speed_value = config.getint("Motion", "speed_value")

# Create a serial connection for motion control
ser = Serial(serial_port, baud_rate, timeout=1)

# Function to return to the origin
def return_to_origin(ser, axis):
    return_origin_command = f"H{axis}0\r".encode()
    ser.write(return_origin_command)
    return_origin_response = ser.readline().decode("utf-8")
    expected_response = f"H{axis}0\rOK\n"
    if expected_response in return_origin_response:
        print(f"Axis {axis} returned to the origin. Response: {return_origin_response}")
        return True
    else:
        print(f"Returning to origin for axis {axis} failed. Expected: {expected_response}, Received: {return_origin_response}")
        return False

# Function to check the connection
def check_connection(ser):
    connection_check_command = "?R\r".encode()
    ser.write(connection_check_command)
    response = ser.readline().decode("utf-8")
    if "?R" in response and "OK" in response:
        print("Connection to motion controller successful")
        return True
    else:
        print("Connection failed.")
        return False


# Function to set the speed
def set_speed(ser, speed_value):
    speed_command = f"V{speed_value}\r".encode()
    ser.write(speed_command)
    speed_response = ser.readline().decode("utf-8")
    if f"V{speed_value}\rOK\n" in speed_response:
        print(f"Speed set to {speed_value} successfully.")
        return True
    else:
        print("Setting speed failed.")
        return False


# Function to move the controller
def move_controller(ser, direction, displacement_value):
    movement_command = f"{direction}+{displacement_value}\r".encode()
    ser.write(movement_command)
    movement_response = ser.readline().decode("utf-8")
    expected_response = f"{direction}+{displacement_value}"
    expected_response = f"{direction}+{displacement_value}"
    if expected_response in movement_response or "OK" in movement_response:
        print(f"Motion in {direction} direction by {displacement_value} successful.")
        return True
    else:
        print(f"Motion in {direction} direction by {displacement_value} failed. Expected: {expected_response}, Received: {movement_response}")
        return False

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Motion controller script")

    subparsers = parser.add_subparsers(dest="command")

    parser_check = subparsers.add_parser("check_connection", help="Check the connection to the motion controller")

    parser_speed = subparsers.add_parser("set_speed", help="Set the speed of the motion controller")
    parser_speed.add_argument("--speed_value", type=int, required=True, help="Speed value to set for the controller")

    parser_move = subparsers.add_parser("move_controller", help="Move the motion controller")
    parser_move.add_argument("--direction", type=str, required=True, help="Direction of the movement")
    parser_move.add_argument("--displacement_value", type=int, required=True, help="Displacement value for the movement")

    parser_return = subparsers.add_parser("return_to_origin", help="Return the motion controller to the origin")
    parser_return.add_argument("--axis", type=str, required=True, help="Axis to return to the origin")

    args = parser.parse_args()

    if args.command == "check_connection":
        check_connection(ser)
    elif args.command == "set_speed":
        set_speed(ser, args.speed_value)
    elif args.command == "move_controller":
        move_controller(ser, args.direction, args.displacement_value)
    elif args.command == "return_to_origin":
        return_to_origin(ser, args.axis)

if __name__ == "__main__":
    main()
